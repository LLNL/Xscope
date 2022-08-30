import math
import numpy
import os

import torch

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
        b = torch.split(b, 5)
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
        self.runs_results = {}
        self.trials_so_far = 0
        self.trials_to_trigger = -1
        self.trials_results = {}
        self.random_results = {}

    def save_trials_to_trigger(self, exp_name: str):
        global trials_to_trigger, trials_so_far
        if trials_to_trigger == -1:
            trials_to_trigger = trials_so_far
            self.trials_results[exp_name] = trials_to_trigger

    def save_results(self, val: float, exp_name: str):
        # Infinity
        if math.isinf(val):
            if exp_name not in self.results.keys():
                if val < 0.0:
                    self.results[exp_name] = [1, 0, 0, 0, 0]
                    self.save_trials_to_trigger(exp_name)
                else:
                    self.results[exp_name] = [0, 1, 0, 0, 0]
                    self.save_trials_to_trigger(exp_name)
            else:
                if val < 0.0:
                    self.results[exp_name][0] += 1
                else:
                    self.results[exp_name][1] += 1

        # Subnormals
        if numpy.isfinite(val):
            if val > -2.22e-308 and val < 2.22e-308:
                if val != 0.0 and val != -0.0:
                    if exp_name not in self.results.keys():
                        if val < 0.0:
                            self.results[exp_name] = [0, 0, 1, 0, 0]
                            self.save_trials_to_trigger(exp_name)
                        else:
                            self.results[exp_name] = [0, 0, 0, 1, 0]
                            self.save_trials_to_trigger(exp_name)
                    else:
                        if val < 0.0:
                            self.results[exp_name][2] += 1
                        else:
                            self.results[exp_name][3] += 1

        if math.isnan(val):
            if exp_name not in self.results.keys():
                self.results[exp_name] = [0, 0, 0, 0, 1]
                self.save_trials_to_trigger(exp_name)
            else:
                self.results[exp_name][4] += 1

    def update_runs_table(self, exp_name: str):
        if exp_name not in self.runs_results.keys():
            self.runs_results[exp_name] = 0
        else:
            self.runs_results[exp_name] += 1

    def are_we_done(self, func, recent_val, exp_name):
        global found_inf_pos, found_inf_neg, found_under_pos, found_under_neg

        # Finding INF+
        if 'max_inf' in func.__name__:
            if found_inf_pos:
                return True
            else:
                if is_inf_pos(recent_val):
                    found_inf_pos = True
                    self.save_results(recent_val, exp_name)
                    return True

        # Finding INF-
        elif 'min_inf' in func.__name__:
            if found_inf_neg:
                return True
            else:
                if is_inf_neg(recent_val):
                    found_inf_neg = True
                    self.save_results(recent_val, exp_name)
                    return True

        # Finding Under-
        elif 'max_under' in func.__name__:
            if found_under_neg:
                return True
            else:
                if is_under_neg(recent_val):
                    found_under_neg = True
                    self.save_results(recent_val, exp_name)
                    return True

        # Finding Under+
        elif 'min_under' in func.__name__:
            if found_under_pos:
                return True
            else:
                if is_under_pos(recent_val):
                    found_under_pos = True
                    self.save_results(recent_val, exp_name)
                    return True
        return False

    def get_results(self):
        return self.results

    def get_random_results(self):
        return self.random_results

    def print_result(self, shared_lib, number_sampling, range_splitting, logger):
        function_key = shared_lib + '|' + number_sampling + '|b_' + range_splitting
        fun_name = os.path.basename(shared_lib)
        print('-------------- Results --------------')
        print(fun_name)
        if len(self.results.keys()) > 0:
            total_exception = 0
            for key in self.results.keys():
                if function_key in key:
                    print('Range:', key.split('|')[0])
                    print('\tINF+:', self.results[key][0])
                    print('\tINF-:', self.results[key][1])
                    print('\tSUB-:', self.results[key][2])
                    print('\tSUB-:', self.results[key][3])
                    print('\tNaN :', self.results[key][4])
                    # print('\tTotal Exception for range {}: {}'.format(key.split('|')[0], sum(results[key]))
                    total_exception += sum(self.results[key])
            print('\tTotal Exception: ', total_exception)
            logger.info('\tTotal Exception: {} '.format(total_exception))
        else:
            print('\tINF+:', 0)
            print('\tINF-:', 0)
            print('\tSUB-:', 0)
            print('\tSUB-:', 0)
            print('\tNaN :', 0)
        print('')

print(bounds("many", 2)[0].shape)