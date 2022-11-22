from functools import partial
import logging

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch import settings as gpt_settings

from botorch.optim import ExpMAStoppingCriterion
from botorch.optim.utils import columnwise_clamp
from botorch.optim.fit import ParameterBounds

from torch.optim import AdamW
from torch.cuda.amp import autocast

from utils import *
from models import ExactGPModel

logging.basicConfig(filename='Xscope.log', level=logging.INFO)
logger = logging.getLogger(__name__)
max_normal = 1e+307

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
                 likelihood_func=GaussianLikelihood(), bounds: Input_bound=None, device=torch.device("cuda")):
        self.eval_func = eval_func
        self.iteration = iteration
        self.batch_size = batch_size
        self.acquisition_function = acquisition_function
        self.bounds_object = bounds
        self.bounds = self.bounds_object.bounds
        self.num_bounds, _, self.dim = self.bounds.shape
        self.device = device
        self.likelihood = likelihood_func.to(device=self.device)
        self.remove_bounds = torch.zeros(self.num_bounds)

        self.results = {}
        self.exception_induced_params = {}
        self.error_types = ["max_inf", "min_inf", "max_under", "min_under", "nan"]
        for type in self.error_types:
            self.results[type] = 0
            self.exception_induced_params[type] = []

        # initialize training data and model
        self.padded_x = torch.ones(self.dim, dtype=dtype, device=self.device)
        self.padded_y = torch.as_tensor(self.eval_func(self.padded_x), device=self.device, dtype=dtype)
        self.model_params_bounds = {}
        self.initialize_data()
        self.initialize_model()
        self.acq = UtilityFunction(self.acquisition_function, kappa=2.5, xi=0.1e-1)

    def initialize_data(self, normalize=False):
        """
        :param
        num_sample: int
            The number of sample to initialize
        :return: Tuple
            A tuple containing the training data
        """
        initial_x = self.bounds_sampler(1)
        if normalize:
            x_min, x_max = initial_x.min(), initial_x.max()
            new_min, new_max = -1e+100, 1e+100
            initial_x = (initial_x - x_min)/(x_max - x_min)*(new_max - new_min) + new_min
        
        self.train_x, self.train_y = self.evaluate_candidates(initial_x)
        if len(self.train_x.shape) == 2:
            self.train_x = self.train_x.unsqueeze(-1)
        assert self.train_y.isnan().sum()==0, "training data result in nan"

    def initialize_model(self, state_dict=None):
        self.likelihood = GaussianLikelihood().to(device=self.device, dtype=dtype)
        self.GP = ExactGPModel(self.train_x, self.train_y, self.likelihood).to(device=self.device, dtype=dtype)
        self.mll = ExactMarginalLogLikelihood(self.likelihood, self.GP)
        for param_name, param in self.mll.model.named_parameters():
            self.model_params_bounds[param_name] = (-10.0, 10.0)
        if state_dict is not None:
            self.GP.load_state_dict(state_dict)
        fit_mll(self.mll, bounds=self.model_params_bounds, options={"disp": False, "lr": 0.05}, approx_mll=True)

    def bounds_sampler(self, num_sample, padding=False):
        lb, ub = self.get_active_bounds()
        num_bounds = lb.shape[0]
        sampler = torch.distributions.uniform.Uniform(lb, ub)
        samples = sampler.rsample((num_sample,)).to(dtype=dtype, device=self.device).view(num_bounds,num_sample, self.dim)
        if padding:
            samples = self.add_padding(samples)
        return samples

    def get_active_bounds(self):
        active_bounds = self.bounds[self.remove_bounds==0]
        lb = active_bounds[:,0,:].unsqueeze(1)
        ub = active_bounds[:,1,:].unsqueeze(1)
        return lb, ub

    def extract_best_candidate(self, candidates):
        with torch.no_grad(), autocast(), gpytorch.settings.fast_pred_var():
            ys = self.acq.forward(self.GP, self.GP.likelihood, candidates).unsqueeze(-1)
            ys = torch.nan_to_num(ys, nan=2.1219957915e-314)
            max_acq, indices = ys.max(dim=1, keepdim=True)
            x_max = torch.gather(candidates, 1, indices.repeat(1,1,self.dim))
        return max_acq, x_max

    def evaluate_candidates(self, candidates, padding=False):
        if padding:
            candidates = self.add_padding(candidates)
        targets = torch.empty((self.num_bounds,1), dtype=dtype, device=self.device)
        for i,x in enumerate(candidates):
            new_candidate_target = self.eval_func(x[0])
            new_candidate_target = torch.as_tensor(new_candidate_target, device=self.device, dtype=dtype)
            if self.check_exception(x, new_candidate_target):
                logger.info( "parameter {} caused floating point error {}".format(x, new_candidate_target))
                x = self.padded_x
                new_candidate_target = self.padded_y
                self.remove_bounds[i] = 1
            candidates[i] = x
            targets[i] = new_candidate_target
        return candidates, targets

    def suggest_new_candidate(self, n_warmup=500, n_samples=5):
        # TODO: add sampling around best point.
        """
            A function to find the maximum of the acquisition function
            It uses a combination of random sampling (cheap) and the 'AdamW'
            optimization method. First by sampling `n_warmup` (500) points per bound at random,
            and then running AdamW from `n_samples` (5) random starting points per bound.
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
        #x_tries has shape: B*n_warmup x D
        x_tries = self.bounds_sampler(n_warmup, padding=True)
        self.likelihood.eval()
        self.GP.eval()
        max_acq, x_max = self.extract_best_candidate(x_tries)

        # Explore the parameter space more throughly
        lb, ub = self.bounds[:,0,:].unsqueeze(1), self.bounds[:,1,:].unsqueeze(1)
        _clamp = partial(columnwise_clamp, lower=lb, upper=ub)
        x_seeds = self.bounds_sampler(n_samples, padding=True)
        clamped_candidates = _clamp(x_seeds).requires_grad_(True)

        to_minimize = lambda x: -self.acq.forward(self.GP, self.GP.likelihood, x)

        optimizer = torch.optim.AdamW([clamped_candidates], lr=1e-5)
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

        clamped_candidates, best_acqs = self.extract_best_candidate(clamped_candidates)
        condition = torch.gt(max_acq, best_acqs)
        best_candidate = torch.where(condition, x_max, clamped_candidates)
        best_candidate = self.add_padding(best_candidate)
        return best_candidate.detach()

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

    def add_padding(self, candidates):
        i = 0
        padded_candidates = torch.empty(self.train_x.shape[0], candidates.shape[1], self.dim, dtype=dtype, device=self.device)
        for index in range(len(padded_candidates)):
            if self.remove_bounds[index] == 1:
                padded_candidates[index] = self.padded_x
            else:
                padded_candidates[index] = candidates[i]
                i += 1
        return padded_candidates

    def train(self):
        # TODO: reimpliment fit function in mixed precision.
        print("Begin BO")
        start_fitting = time.time()
        for i in range(self.iteration):
            if i % self.batch_size == 0 and i != 0:
                old_state_dict = self.mll.model.state_dict()
                self.initialize_model(state_dict=old_state_dict)
            new_candidates = self.suggest_new_candidate()
            new_candidates, new_targets = self.evaluate_candidates(new_candidates, padding=True)
            self.train_x = torch.cat([self.train_x, new_candidates], dim=1)
            self.train_y = torch.cat([self.train_y, new_targets], dim=1)
            assert self.train_x.shape[0] == self.train_y.shape[0], f"shape mismatch, got {self.train_x.shape[0]} for training data but {self.train_y.shape[0]} for testing data"

        print("Fitting time: ", time.time() - start_fitting)

        # for type in self.error_types:
        #     self.exception_induced_params[type] = torch.cat(self.exception_induced_params[type]).flatten()
