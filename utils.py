import math
import numpy
import torch
from torch.linalg import norm
from os.path import isfile
import time
import pandas as pd

max_normal = 1e+307
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

def validate_output(output, new_max):
    if numpy.isposinf(output):
        output = new_max
    elif numpy.isneginf(output):
        output = -new_max
    elif numpy.isnan(output):
        output = new_max
    return output


def bounds(split, num_input, input_type="fp"):
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
        b = torch.as_tensor(b, dtype=dtype, device=device)

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
        b = torch.as_tensor(b, dtype=dtype, device=device)

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
                b.append(torch.tensor([r1], dtype=dtype, device=device).squeeze(0))
        elif num_input == 2:
            for r1 in ranges:
                for r2 in ranges:
                    bound = torch.transpose(torch.tensor([r1,r2], dtype=dtype, device=device).squeeze(),0,1)
                    b.append(bound)
        else:
            for r1 in ranges:
                for r2 in ranges:
                    bound = torch.transpose(torch.tensor([r1,r2,r2], dtype=dtype, device=device).squeeze(),0,1)
                    b.append(bound)
        b = torch.stack(b, dim=0)
        # b = torch.split(b, 10)
    return b


def is_inf_pos(val):
    if math.isinf(val):
        return val > 0.0
    return False


def is_inf_neg(val):
    if math.isinf(val):
        return val < 0.0
    return False


def is_under_pos(val):
    if numpy.isfinite(val):
        if val > 0.0 and val < 2.22e-308:
            return True
    return False


def is_under_neg(val):
    if numpy.isfinite(val):
        if val < 0.0 and val > -2.22e-308:
            return True
    return False


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

    def forward(self,gp, likelihood, x, y_max):
        if self.kind == 'ucb':
            return self._ucb(gp, likelihood, x, self.kappa)
        if self.kind == 'ei':
            return self._ei(gp, likelihood, x, y_max, self.xi)
        if self.kind == 'poi':
            return self._poi(gp, likelihood, x, y_max, self.xi)

    @staticmethod
    def _ucb(gp, likelihood, x, kappa):
        output = likelihood(gp.forward(x))
        mean, std = output.mean, torch.sqrt(output.var)
        return mean + kappa * std

    @staticmethod
    def _ei(gp, likelihood, x, y_max, xi):
        output = likelihood(gp.forward(x))
        mean, std = output.mean, torch.sqrt(output.var)
        a = (mean - y_max - xi)
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _poi(gp, likelihood, x, y_max, xi):
        output = likelihood(gp.forward(x))
        mean, std = output.mean, torch.sqrt(output.var)
        z = (mean - y_max - xi)/std
        return norm.cdf(z)


