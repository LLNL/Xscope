from functools import partial
import logging

import gpytorch
from gpytorch.likelihoods import GaussianLikelihood, likelihood_list
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from gpytorch import settings as gpt_settings

from botorch.optim import ExpMAStoppingCriterion
from botorch.optim.utils import columnwise_clamp, _get_extra_mll_args, _expand_bounds
from botorch.optim.fit import fit_gpytorch_torch, ParameterBounds

from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

from utils import *

logging.basicConfig(filename='Xscope.log', level=logging.INFO)
logger = logging.getLogger(__name__)
max_normal = 1e+307


class ExactGPModel(gpytorch.models.ExactGP):
    """
    Class for Gaussian Process Model
    Attributes
    ----------
    train_x: a Torch tensor with shape n x d
        The training data
    train_x: a Torch tensor with shape n x 1
        The label
    likelihood : The likelihood function
        A function that approximate the likelihood of the GP given new observed datapoint
    Methods
    -------
    forward
    """

    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.MaternKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, batch_shape):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([2]))
        self.covar_module = gpytorch.kernels.MaternKernel(
            batch_shape=batch_shape
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


class BaysianOptimization():
    """
    Class for Baysian Optimization for inputs that trigger FP-exceptions Model
    Attributes
    ----------
    eval_func: GPU function
        The function that are being tested on
    iteration: int
        How many time will the BO be run for
    batch_size: int
        How many time we run the GP before we update it
    acquisition_function: string
        A string indicating which acquisition function to use
        As of right now, we only accept UCB, IP, EI.
    likelihood : The likelihood function
        A function that approximate the likelihood of the GP given new observed datapoint
    bounds: a Tensor with shape 2xd
        The bound of each of the input parameter
    initial_sample: int
        Number of initial data points to sample.
    device:
        The device that the BO will be run on
    Methods
    -------
    initialize_data
    """

    def __init__(self, eval_func, iteration=25, batch_size=5, acquisition_function='ucb',
                 likelihood_func=GaussianLikelihood(), bounds=None, initial_sample=2, device=torch.device("cuda")):
        self.eval_func = eval_func
        self.iteration = iteration
        self.batch_size = batch_size
        self.acquisition_function = acquisition_function
        self.bounds = bounds

        self.lower_bounds = torch.select(self.bounds, 1, 0).unsqueeze(1)
        self.upper_bounds = torch.select(self.bounds, 1, 1).unsqueeze(1)
        self.device = device
        self.likelihood = likelihood_func.to(device=self.device)

        self.results = {}
        self.exception_induced_params = {}
        self.error_types = ["max_inf", "min_inf", "max_under", "min_under", "nan"]
        for type in self.error_types:
            self.results[type] = 0
            self.exception_induced_params[type] = []

        # initialize training data and model
        self.train_x, self.train_y = self.initialize_data(initial_sample)
        self.initialize_model()
        self.acq = UtilityFunction(self.acquisition_function, kappa=2.5, xi=0.1e-1)
        self.optim = AdamW

    def initialize_data(self, num_sample=5):
        """
        :param
        num_sample: int
            The number of sample to initialize
        :return: Tuple
            A tuple containing the training data
        """
        init_sampler = self.bounds_sampler(num_sample)
        initial_X = init_sampler.sample().to(device=device, dtype=dtype)
        initial_X = initial_X.flatten(0,1)
        initial_Y = torch.zeros(initial_X.shape[0], device=device, dtype=dtype)
        for i, x in enumerate(initial_X):
            new_cadidate_target = self.eval_func(x)
            new_cadidate_target = torch.as_tensor(new_cadidate_target, device=self.device, dtype=dtype)
            if self.check_exception(x, new_cadidate_target.detach()):
                logger.info( "parameter {} caused floating point error {}".format(x, new_cadidate_target.detach()))
                # if we detect error at initialization stage, we log the error and try again
                return self.initialize_data(num_sample)
            initial_Y[i] = new_cadidate_target
        return initial_X, initial_Y

    def initialize_model(self, state_dict=None):
        self.likelihood = GaussianLikelihood()
        self.GP = ExactGPModel(self.train_x, self.train_y, self.likelihood)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.GP)
        if state_dict is not None:
            self.GP.load_state_dict(state_dict)

    def bounds_sampler(self, num_sample):
        lbound = self.lower_bounds.expand(-1, num_sample, -1)
        ubound = self.upper_bounds.expand(-1, num_sample, -1)
        return torch.distributions.uniform.Uniform(lbound, ubound)

    def suggest_new_candidate(self, n_warmup=10000, n_samples=10):
        # TODO: add sampling around best point.
        # TODO: autocast during warm up.
        """
            A function to find the maximum of the acquisition function
            It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
            optimization method. First by sampling `n_warmup` (1e5) points at random,
            and then running L-BFGS-B from `n_iter` (250) random starting points.
            Parameters
            ----------
            :param n_warmup:
                number of times to randomly sample the acquisition function
            :param n_samples:
                number of samples to try
            Returns
            -------
            :return: x_max, The arg max of the acquisition function.
            """

        # Warm up with random points
        batch_size = self.lower_bounds.shape[0]
        dim_size = self.lower_bounds.shape[-1]
        warmup_sampler = self.bounds_sampler(n_warmup)
        #x_tries has shape: B*n_warmup x D
        x_tries = warmup_sampler.sample().flatten(0,1).to(device=device, dtype=dtype)
        x_tries_flatten = x_tries.flatten(0,1)
        self.likelihood.eval()
        self.GP.eval()
        with torch.no_grad(), autocast(), gpytorch.settings.fast_pred_var():
            ys = self.acq.forward(self.GP, self.GP.likelihood, x_tries_flatten)
            ys = torch.reshape(ys,(batch_size, n_warmup))
            max_acq, indices = ys.max(dim=1)
            x_max = x_tries[range(batch_size), indices,]

        # Explore the parameter space more throughly
        explore_sampler = self.bounds_sampler(n_samples)

        _clamp = partial(columnwise_clamp, lower=self.lower_bounds, upper=self.upper_bounds)
        x_seeds = explore_sampler.sample().flatten(0,1).to(device=device, dtype=dtype)
        clamped_candidates = _clamp(x_seeds).requires_grad_(True)

        to_minimize = lambda x: -self.acq.forward(self.GP, self.GP.likelihood, x)

        optimizer = torch.optim.AdamW([clamped_candidates], lr=0.025)
        i = 0
        stop = False
        stopping_criterion = ExpMAStoppingCriterion(maxiter=150)

        while not stop:
            i += 1
            with torch.no_grad():
                X = _clamp(clamped_candidates).requires_grad_(True)

            with autocast(), gpytorch.settings.fast_pred_var():
                loss = to_minimize(X).sum()

            grad_params = torch.autograd.grad(loss, X)[0]

            def assign_grad():
                optimizer.zero_grad()
                clamped_candidates.grad = grad_params
                return loss

            optimizer.step(assign_grad)
            if clamped_candidates.isnan().sum() > 0:
                clamped_candidates = X
                break
            stop = stopping_criterion.evaluate(fvals=loss.detach())

        clamped_candidates = _clamp(clamped_candidates)
        with torch.no_grad(), autocast(), gpytorch.settings.fast_pred_var():
            batch_acquisition = -to_minimize(clamped_candidates)
            batch_acquisition = torch.reshape(batch_acquisition, (batch_size, n_samples))

        best_acqs, indices = batch_acquisition.max(dim=1)
        clamped_candidates = torch.reshape(clamped_candidates, (batch_size, n_samples, dim_size))
        clamped_candidates = clamped_candidates[range(clamped_candidates.shape[0]), indices,]
        best_candidate = torch.zeros_like(clamped_candidates)
        compare = torch.gt(max_acq, best_acqs)
        for index, result in enumerate(compare):
            if result:
                best_candidate[index] = x_max[index]
            else:
                best_candidate[index] = clamped_candidates[index]
        return best_candidate.flatten(0,1).unsqueeze(1).detach()

    def check_exception(self, param, val):
        # TODO: check exceptions in batch.
        # TODO: Flatten the the bounds + num_samples (B X N X D -> B*N X D) and run the function normally. Tested on all bound at once. Remove bounds that encounter exception.
        """
            A function to check if the value returned by the GPU function is an FP exception. It also updates the result
            table if an exception is caught
            ----------
            :param param:
                The input parameter to the evaluation function
            :param val:
                The return value of the evaluation function
            Returns
            -------
            :return: a boolean indicating if the value is an FP exception or not
            """
        # Infinity
        if torch.isinf(val):
            if val < 0.0:
                self.results["min_inf"] += 1
                self.exception_induced_params["min_inf"].append(param)
                # self.save_trials_to_trigger(exp_name)
            else:
                self.results["max_inf"] += 1
                self.exception_induced_params["max_inf"].append(param)
                # self.save_trials_to_trigger(exp_name)
            return True

        # Subnormals
        if torch.isfinite(val):
            if val > -2.22e-308 and val < 2.22e-308:
                if val != 0.0 and val != -0.0:
                    if val < 0.0:
                        self.results["min_under"] += 1
                        self.exception_induced_params["min_under"].append(param)
                    else:
                        self.results["max_under"] += 1
                        self.exception_induced_params["max_under"].append(param)
                    return True

        if torch.isnan(val):
            self.results["nan"] += 1
            self.exception_induced_params["nan"].append(param)
            return True
        return False

    def train(self):
        # TODO: reimpliment fit function in mixed precision.
        print("Begin BO")
        start_fitting = time.time()
        fit_gpytorch_torch(self.mll, options={"disp": False})
        for i in range(self.iteration):
            # Set the gradients from previous iteration to zero
            if i % self.batch_size == 0 and i != 0:
                self.initialize_model(state_dict=self.mll.model.state_dict())
                fit_gpytorch_torch(self.mll, options={"disp": False})

            new_candidates = self.suggest_new_candidate()
            for candidate in new_candidates:
                new_candidate_target = self.eval_func(candidate.detach())
                new_candidate_target = torch.as_tensor([new_candidate_target])
                if self.check_exception(candidate, new_candidate_target.detach()):
                    logger.info(
                        "parameter {} caused floating point error {}".format(candidate, new_candidate_target.detach()))
                    continue
                self.train_x = torch.cat([self.train_x, candidate], dim=1)
                self.train_y = torch.cat([self.train_y, new_candidate_target], dim=1)
        print("Fitting time: ", time.time() - start_fitting)

        # for type in self.error_types:
        #     self.exception_induced_params[type] = torch.cat(self.exception_induced_params[type]).flatten()
