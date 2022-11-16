import math

import gpytorch
import torch
from torch.distributions import Normal
from os.path import isfile
import time
from time import monotonic
import pandas as pd
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch import settings as gpt_settings

from botorch.optim import ExpMAStoppingCriterion
from botorch.optim.utils import  _get_extra_mll_args
from botorch.optim.fit import ParameterBounds
from botorch.optim.utils import (
    _filter_kwargs,
    _get_extra_mll_args,
    create_name_filter,
)

from torch.optim import AdamW
from torch.cuda.amp import autocast
from torch.optim.optimizer import Optimizer
from torch import Tensor
from torch.cuda.amp import autocast

import jax.numpy as jnp



from typing import (
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

class OptimizationIteration(NamedTuple):
    itr: int
    fun: float
    time: float


max_normal = 1e+307
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

from jax.config import config
config.update("jax_enable_x64", True)

class Input_bound():
    def __init__(self, split="many", num_input=1, input_type="fp", device = torch.device("cuda")) -> None:
        self.device = device
        self.bounds = self.generate_bounds(split, num_input, input_type)
        self.num_bounds, _, self.dim = self.bounds.shape
        self.padded_value = torch.ones(self.dim, dtype=dtype, device=self.device)

    def generate_bounds(self, split, num_input, input_type="fp"):
        b = []
        upper_lim = max_normal
        lower_lim = -max_normal
        if input_type == "exp":
            upper_lim = 307
            lower_lim = -307
        if split == "whole":
            if num_input == 1:
                b.append([[lower_lim], [upper_lim]])
            elif num_input == 2:
                b.append([[lower_lim, lower_lim], [upper_lim, upper_lim]])
            else:
                b.append([[lower_lim, lower_lim, lower_lim], [upper_lim, upper_lim, upper_lim]])
            b = torch.as_tensor(b, dtype=dtype, device=self.device)

        elif split == "two":
            if num_input == 1:
                b.append([[lower_lim], [0]])
                b.append([[0], [upper_lim]])
            elif num_input == 2:
                b.append([[lower_lim, lower_lim], [0, 0]])
                b.append([[0, 0],[upper_lim, upper_lim]])
            else:
                b.append([[lower_lim, lower_lim, lower_lim], [0, 0, 0]])
                b.append([[0, 0,0], [upper_lim, upper_lim, upper_lim]])
            b = torch.as_tensor(b, dtype=dtype, device=self.device)

        else:
            limits = [0.0, 1e-307, 1e-100, 1e-10, 1e-1, 1e0, 1e+1, 1e+10, 1e+100, 1e+307]
            ranges = []
            if input_type == "exp":
                for i in range(len(limits) - 1):
                    x = limits[i]
                    y = limits[i + 1]
                    t = (min(x, y), max(x, y))
                    ranges.append(t)
            else:
                for i in range(len(limits) - 1):
                    x = limits[i]
                    y = limits[i + 1]
                    t = [[min(x, y)], [max(x, y)]]
                    ranges.append(t)
                    x = -limits[i]
                    y = -limits[i + 1]
                    t = [[min(x, y)], [max(x, y)]]
                    ranges.append(t)
            if num_input == 1:
                for r1 in ranges:
                    b.append(torch.tensor([r1], dtype=dtype, device=self.device).squeeze(0))
            elif num_input == 2:
                for r1 in ranges:
                    for r2 in ranges:
                        bound = torch.transpose(torch.tensor([r1,r2], dtype=dtype, device=self.device).squeeze(),0,1)
                        b.append(bound)
            else:
                for r1 in ranges:
                    for r2 in ranges:
                        bound = torch.transpose(torch.tensor([r1,r2,r2], dtype=dtype, device=self.device).squeeze(),0,1)
                        b.append(bound)
            b = torch.stack(b, dim=0)
        print("number of bounds to test: ", b.shape)
        return b

    def generate_bounds_np(self, split, num_input, input_type="fp"):
        b = []
        upper_lim = max_normal
        lower_lim = -max_normal
        if input_type == "exp":
            upper_lim = 307
            lower_lim = -307
        if split == "whole":
            if num_input == 1:
                b.append([[lower_lim], [upper_lim]])
            elif num_input == 2:
                b.append([[lower_lim, lower_lim], [upper_lim, upper_lim]])
            else:
                b.append([[lower_lim, lower_lim, lower_lim], [upper_lim, upper_lim, upper_lim]])
            b = jnp.asarray(b, dtype=jnp.float64)

        elif split == "two":
            if num_input == 1:
                b.append([[lower_lim], [0]])
                b.append([[0], [upper_lim]])
            elif num_input == 2:
                b.append([[lower_lim, lower_lim], [0, 0]])
                b.append([[0, 0],[upper_lim, upper_lim]])
            else:
                b.append([[lower_lim, lower_lim, lower_lim], [0, 0, 0]])
                b.append([[0, 0,0], [upper_lim, upper_lim, upper_lim]])
            b = jnp.asarray(b, dtype=jnp.float64)

        else:
            limits = jnp.array([0.0, 1e-307, 1e-100, 1e-10, 1e-1, 1e0, 1e+1, 1e+10, 1e+100, 1e+307])
            ranges = []
            if input_type == "exp":
                for i in range(len(limits) - 1):
                    x = limits[i]
                    y = limits[i + 1]
                    t = (min(x, y), max(x, y))
                    ranges.append(t)
            else:
                for i in range(len(limits) - 1):
                    x = limits[i]
                    y = limits[i + 1]
                    t = [[min(x, y)], [max(x, y)]]
                    ranges.append(t)
                    x = -limits[i]
                    y = -limits[i + 1]
                    t = [[min(x, y)], [max(x, y)]]
                    ranges.append(t)
            if num_input == 1:
                for r1 in ranges:
                    b.append(jnp.array(r1, dtype=jnp.float64))
            elif num_input == 2:
                for r1 in ranges:
                    for r2 in ranges:
                        bound = jnp.transpose(jnp.array([r1,r2], dtype=jnp.float64).squeeze(),(1,0))
                        b.append(bound)
            else:
                for r1 in ranges:
                    for r2 in ranges:
                        bound = jnp.transpose(jnp.array([r1,r2, r2], dtype=jnp.float64).squeeze(),(1,0))
                        b.append(bound)
            b = jnp.stack(b, axis=0)
        print("number of bounds to test: ", b.shape)
        return b
    
    def bounds_sampler(self, num_sample, padding=False):
        lb, ub = self.get_active_bounds()
        num_bounds = lb.shape[0]
        sampler = torch.distributions.uniform.Uniform(lb, ub)
        samples = sampler.rsample((num_sample,)).to(dtype=dtype, device=self.device).view(num_bounds, num_sample, self.dim)
        if padding:
            padding_shape = torch.empty(self.num_bounds, num_sample, self.dim)
            samples = self.add_padding(samples, padding_shape)
        return samples

    def add_padding(self, candidates, padding_shape):
        i = 0
        padded_candidates = torch.empty_like(padding_shape, dtype=dtype, device=self.device)
        for index in range(len(padded_candidates)):
            if self.remove_bounds[index] == 1:
                padded_candidates[index] = self.padded_value
            else:
                padded_candidates[index] = candidates[i]
                i += 1
        return padded_candidates
        
def fit_mll(
    mll: MarginalLogLikelihood,
    bounds: Optional[ParameterBounds] = None,
    optimizer_cls: Optimizer = AdamW,
    options: Optional[Dict[str, Any]] = None,
    track_iterations: bool = False,
    approx_mll: bool = False,
) -> Tuple[MarginalLogLikelihood, Dict[str, Union[float, List[OptimizationIteration]]]]:
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

    Example:
        >>> gp = SingleTaskGP(train_X, train_Y)
        >>> mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        >>> mll.train()
        >>> fit_gpytorch_torch(mll)
        >>> mll.eval()
    """
    optim_options = {"maxiter": 100, "disp": True, "lr": 0.05}
    optim_options.update(options or {})
    exclude = optim_options.pop("exclude", None)
    if exclude is None:
        mll_params = list(mll.parameters())
    else:
        mll_params = [
            v for k, v in filter(create_name_filter(exclude), mll.named_parameters())
        ]

    optimizer = optimizer_cls(
        params=[{"params": mll_params}],
        **_filter_kwargs(optimizer_cls, **optim_options),
    )

    # get bounds specified in model (if any)
    bounds_: ParameterBounds = {}
    if hasattr(mll, "named_parameters_and_constraints"):
        for param_name, _, constraint in mll.named_parameters_and_constraints():
            if constraint is not None and not constraint.enforced:
                bounds_[param_name] = constraint.lower_bound, constraint.upper_bound

    # update with user-supplied bounds (overwrites if already exists)
    if bounds is not None:
        bounds_.update(bounds)

    iterations = []
    t1 = monotonic()

    param_trajectory: Dict[str, List[Tensor]] = {
        name: [] for name, param in mll.named_parameters()
    }
    loss_trajectory: List[float] = []
    i = 0
    stop = False
    stopping_criterion = ExpMAStoppingCriterion(
        **_filter_kwargs(ExpMAStoppingCriterion, **optim_options)
    )
    train_inputs, train_targets = mll.model.train_inputs, mll.model.train_targets
    while not stop:
        optimizer.zero_grad()
        with autocast(), gpt_settings.fast_computations(log_prob=approx_mll):
            output = mll.model(*train_inputs)
            # we sum here to support batch mode
            args = [output, train_targets] + _get_extra_mll_args(mll)
            loss = -mll(*args).sum()
            loss.backward()
        loss_trajectory.append(loss.item())
        for name, param in mll.named_parameters():
            param.grad.data.clamp_(-1e+50, 1e+50)
            param_trajectory[name].append(param.detach().clone())

        if track_iterations:
            iterations.append(OptimizationIteration(i, loss.item(), monotonic() - t1))

        optimizer.step()
        # project onto bounds:
        if bounds_:
            for pname, param in mll.named_parameters():
                if pname in bounds_:
                    param.data = param.data.clamp(*bounds_[pname])
        i += 1
        stop = stopping_criterion.evaluate(fvals=loss.detach())
    info_dict = {
        "fopt": loss_trajectory[-1],
        "wall_time": monotonic() - t1,
        "iterations": iterations,
    }
    return mll, info_dict

class ResultLogger:
    def __init__(self):
        self.results = {}
        self.random_results = {}
        self.exception_induced_params = {}
        self.bounds_list = []
        self.total_error_per_bound = []
        self.start = 0
        self.execution_time = 0

        self.funcs = ["max_inf", "min_inf", "max_under", "min_under", "nan"]

        for type in self.funcs:
            self.results[type] = 0
            self.exception_induced_params[type] = []

    def start_time(self):
        self.start = time.time()

    def log_time(self):
        self.execution_time = time.time() - self.start

    def log_result(self, bo_errors):
        error_count = 0
        for type in self.funcs:
            self.results[type] += bo_errors[type]
            # result_logger.exception_induced_params[type] += bo.exception_induced_params[type]
            error_count += self.results[type]
        self.total_error_per_bound.append(error_count)

    def summarize_result(self, func_name):
        total_exception = 0
        bgrt_bo_data = {'Function': [func_name]}
        for type in self.funcs:
            print('\t' + type + ": ", self.results[type])
            bgrt_bo_data[type] = [self.results[type]]
            total_exception += self.results[type]

        print('\tTotal Exception: ', total_exception)
        bgrt_bo_data.update({'Total Exception': [total_exception],
                             'Execution Time': [self.execution_time]})
        bgrt_interval_data = {}
        bgrt_interval_data['Function'] = [func_name]
        for bound, total_error in zip(self.bounds_list, self.total_error_per_bound):
            bgrt_interval_data[bound] = [total_error]

        bgrt_bo_df = pd.DataFrame(bgrt_bo_data)
        bgrt_interval_df = pd.DataFrame(bgrt_interval_data)

        return bgrt_bo_df, bgrt_interval_df

    def write_result_to_file(self, file_name, data):
        if isfile(file_name):
            data.to_csv(file_name, mode='a', index=False, header=False)
        else:
            data.to_csv(file_name, index=False)

"""
This code is inspired by this github: https://github.com/fmfn/BayesianOptimization
"""

class UtilityFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, kind, kappa, xi, kappa_decay=1, kappa_decay_delay=0):
        self.kappa = kappa
        self._kappa_decay = kappa_decay
        self._kappa_decay_delay = kappa_decay_delay

        self.xi = xi

        self._iters_counter = 0

        if kind not in ['ucb', 'ei', 'poi']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def update_params(self):
        self._iters_counter += 1

        if self._kappa_decay < 1 and self._iters_counter > self._kappa_decay_delay:
            self.kappa *= self._kappa_decay

    def forward(self,gp, likelihood, x, y_max= None):
        if self.kind == 'ucb':
            return self._ucb(gp, likelihood, x, self.kappa)
        if self.kind == 'ei':
            return self._ei(gp, likelihood, x, y_max, self.xi)
        if self.kind == 'poi':
            return self._poi(gp, likelihood, x, y_max, self.xi)

    @staticmethod
    def _ucb(gp, likelihood, x, kappa):
        with gpytorch.settings.fast_pred_var():
            output = likelihood(gp(x))
        mean, std = output.mean, torch.sqrt(output.variance)
        return mean + kappa * std

    @staticmethod
    def _ei(gp, likelihood, x, y_max, xi):
        with gpytorch.settings.fast_pred_var():
            output = likelihood(gp(x))
        mean, std = output.mean, torch.sqrt(output.variance)
        a = (mean - y_max - xi)
        z = a / std
        if not torch.isfinite(z).all():
            return a + std
        norm = Normal(torch.tensor([0.0]).to(device=device, dtype=dtype), torch.tensor([1.0]).to(device=device, dtype=dtype))
        pdf = 1/torch.sqrt(torch.tensor(2.0*math.pi).to(device=device, dtype=dtype)) * torch.exp(-z**2/2)
        return a * norm.cdf(z) + std * pdf

    @staticmethod
    def _poi(gp, likelihood, x, y_max, xi):
        with gpytorch.settings.fast_pred_var():
            output = likelihood(gp(x))
        mean, std = output.mean, torch.sqrt(output.variance)
        z = (mean - y_max - xi)/std
        norm = Normal(torch.tensor([0.0]).to(device=device, dtype=dtype), torch.tensor([1.0]).to(device=device, dtype=dtype))
        return norm.cdf(z)
