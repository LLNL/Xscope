from functools import partial
import logging

import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch import settings as gpt_settings

from botorch.optim import ExpMAStoppingCriterion
from botorch.optim.utils import columnwise_clamp, _get_extra_mll_args
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
                 likelihood_func=GaussianLikelihood(), bounds=None, initial_sample=10, device=torch.device("cuda")):
        self.eval_func = eval_func
        self.iteration = iteration
        self.batch_size = batch_size
        self.acquisition_function = acquisition_function
        self.bounds = bounds
        self.device = device
        self.likelihood = likelihood_func.to(device=self.device)

        self.results = {}
        self.exception_induced_params = {}
        self.error_types = ["max_inf", "min_inf", "max_under", "min_under", "nan"]
        for type in self.error_types:
            self.results[type] = 0
            self.exception_induced_params[type] = []

        self.sampler = torch.distributions.uniform.Uniform(self.bounds[0], self.bounds[1])

        # initialize training data and model
        self.train_x, self.train_y = self.initialize_data(initial_sample)
        self.y_max = self.train_y.max()
        self.initialize_model()
        self.acq = UtilityFunction(self.acquisition_function, kappa=2.5, xi=0.1e-1)
        self.optim = AdamW

    def initialize_model(self, state_dict=None):
        self.GP = ExactGPModel(self.train_x, self.train_y, self.likelihood).to(device=self.device, dtype=dtype)
        self.mll = ExactMarginalLogLikelihood(self.GP.likelihood, self.GP)
        if state_dict is not None:
            self.GP.load_state_dict(state_dict)

    def initialize_data(self, num_sample=5):
        """
        :param
        eval_func: GPU function
            The function that are being tested on
        num_sample: int
            The number of sample to initialize
        :return: Tuple
            A tuple containing the training data
        """
        initial_X = self.sampler.sample((num_sample,)).to(device=device, dtype=dtype)
        initial_Y = torch.zeros(num_sample).to(device=device, dtype=dtype)
        for index, x in enumerate(initial_X):
            new_cadidate_target = self.eval_func(x)
            new_cadidate_target = torch.as_tensor(new_cadidate_target, device=self.device, dtype=dtype)
            if self.check_exception(x, new_cadidate_target):
                logger.info("parameter {} caused floating point error {}".format(x, new_cadidate_target))
                continue

        return initial_X, initial_Y

    def suggest_new_candidate(self,n_warmup=10000, n_samples=25):
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
        x_tries = self.sampler.sample((n_warmup,)).to(device=device, dtype=dtype)
        self.likelihood.eval()
        self.GP.eval()
        with torch.no_grad():
            ys = self.acq.forward(self.GP, self.GP.likelihood, x_tries, self.y_max)
            x_max = x_tries[ys.argmax()]
            max_acq = ys.max()

        # Explore the parameter space more throughly

        _clamp = partial(columnwise_clamp, lower=self.bounds[0], upper=self.bounds[1])
        x_seeds = self.sampler.sample((n_samples,)).to(device=device, dtype=dtype)
        clamped_candidates = _clamp(x_seeds).requires_grad_(True)

        to_minimize = lambda x: -self.acq.forward(self.GP, self.GP.likelihood, x, y_max=self.y_max)

        optimizer = torch.optim.AdamW([clamped_candidates], lr=0.025)

        scaler = GradScaler()

        i = 0
        stop = False
        stopping_criterion = ExpMAStoppingCriterion(maxiter=150)

        while not stop:
            i += 1
            with torch.no_grad():
                X = _clamp(clamped_candidates).requires_grad_(True)

            with autocast():
                loss = to_minimize(X).sum()
            # if not torch.isfinite(loss).all():
            #     continue
            scaled_grad = torch.autograd.grad(outputs=scaler.scale(loss),
                                              inputs=X,
                                              create_graph=True)[0]

            inv_scale = 1. / scaler.get_scale() if scaler.get_scale() != 0 else 1
            grad_params = scaled_grad * inv_scale

            optimizer.zero_grad()
            clamped_candidates.grad = grad_params

            scaler.step(optimizer)
            scaler.update()
            stop = stopping_criterion.evaluate(fvals=loss.detach())

        clamped_candidates = _clamp(clamped_candidates)
        with torch.no_grad():
            batch_acquisition = -to_minimize(clamped_candidates)

        best = torch.argmax(batch_acquisition.view(-1), dim=0)
        if batch_acquisition[best] < max_acq:
            best_candidate = x_max
        else:
            best_candidate = clamped_candidates[best]
        return best_candidate.unsqueeze(0).detach()

    def check_exception(self, param, val):
        # TODO: check exceptions in batch.
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

    def fit_mll(self, iters_to_accumulate = 5):
        r"""Fit a gpytorch model by maximizing MLL with a torch optimizer.

            The model and likelihood in mll must already be in train mode.
            Note: this method requires that the model has `train_inputs` and `train_targets`.

            Args:
                mll: MarginalLogLikelihood to be maximized.
                bounds: A ParameterBounds dictionary mapping parameter names to tuples
                    of lower and upper bounds. Bounds specified here take precedence
                    over bounds on the same parameters specified in the constraints
                    registered with the module.
                optimizer_cls: Torch optimizer to use. Must not require a closure.
                options: options for model fitting. Relevant options will be passed to
                    the `optimizer_cls`. Additionally, options can include: "disp"
                    to specify whether to display model fitting diagnostics and "maxiter"
                    to specify the maximum number of iterations.
                track_iterations: Track the function values and wall time for each
                    iteration.
                approx_mll: If True, use gpytorch's approximate MLL computation (
                    according to the gpytorch defaults based on the training at size).
                    Unlike for the deterministic algorithms used in fit_gpytorch_scipy,
                    this is not an issue for stochastic optimizers.

            Returns:
                2-element tuple containing
                - mll with parameters optimized in-place.
                - Dictionary with the following key/values:
                "fopt": Best mll value.
                "wall_time": Wall time of fitting.
                "iterations": List of OptimizationIteration objects with information on each
                iteration. If track_iterations is False, will be empty.
            """
        mll_params = list(self.mll.parameters())
        optimizer = self.optim(
            params=[{"params": mll_params}], lr=0.05
        )

        # get bounds specified in model (if any)
        bounds_: ParameterBounds = {}
        if hasattr(self.mll, "named_parameters_and_constraints"):
            for param_name, _, constraint in self.mll.named_parameters_and_constraints():
                if constraint is not None and not constraint.enforced:
                    bounds_[param_name] = constraint.lower_bound, constraint.upper_bound

        stop = False
        stopping_criterion = ExpMAStoppingCriterion()
        train_inputs, train_targets = self.mll.model.train_inputs, self.mll.model.train_targets

        scaler = GradScaler()
        i = 0

        while not stop:
            optimizer.zero_grad()
            with gpt_settings.fast_computations(log_prob=True), autocast():
                output = self.mll.model(*train_inputs)
                # we sum here to support batch mode
                args = [output, train_targets] + _get_extra_mll_args(self.mll)
                loss = -self.mll(*args).sum()/iters_to_accumulate

            scaler.scale(loss).backward()
            if (i + 1) % iters_to_accumulate == 0:
                # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            stop = stopping_criterion.evaluate(fvals=loss.detach())
            i += 1

    def train(self):
        # TODO: reimpliment fit function in mixed precision and scaler.
        print("Begin BO for bound {}".format(self.bounds))
        self.fit_mll()
        for i in range(self.iteration):
            # Set the gradients from previous iteration to zero
            if i%self.batch_size==0 and i!=0:
                self.initialize_model(state_dict=self.mll.model.state_dict())
                self.fit_mll()

            start_candidate_suggest = time.time()
            new_candidate = self.suggest_new_candidate()
            print("Time used to find new candidate: ", time.time() - start_candidate_suggest)
            new_candidate_target = self.eval_func(new_candidate)
            new_candidate_target = torch.as_tensor([new_candidate_target]).to(self.train_y)
            print("new target: ", new_candidate_target)
            if self.check_exception(new_candidate, new_candidate_target):
                logger.info("parameter {} caused floating point error {}".format(new_candidate, new_candidate_target))
                break
            self.train_x = torch.cat([self.train_x, new_candidate])
            self.train_y = torch.cat([self.train_y, new_candidate_target])

        # for type in self.error_types:
        #     self.exception_induced_params[type] = torch.cat(self.exception_induced_params[type]).flatten()
