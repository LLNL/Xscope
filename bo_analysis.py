#!//usr/bin/env python3
import logging
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from functools import partial
from utils import *
from test_function import *

# verbose = False
verbose = True
CUDA_LIB = ''
bo_iterations = 25  # number of iteration
logging.basicConfig(filename='Xscope.log', level=logging.INFO)
logger = logging.getLogger(__name__)

# ----- Status variables ------
found_inf_pos = False
found_inf_neg = False
found_under_pos = False
found_under_neg = False
# -----------------------------

def initialize():
    global found_inf_pos, found_inf_neg, found_under_pos, found_under_neg
    found_inf_pos = False
    found_inf_neg = False
    found_under_pos = False
    found_under_neg = False

def set_max_iterations(n: int):
    global bo_iterations
    bo_iterations = n

test_func = TestFunction(CUDA_LIB)
result_logger = ResultLogger()

# ----------------------------------------------------------------------------
# Results Checking
# ----------------------------------------------------------------------------

def run_optimizer(bounds, func, new_max, exp_name):
    global trials_to_trigger, trials_so_far
    num_fail = 0
    trials_so_far = 0
    trials_to_trigger = -1

    # if are_we_done(func, 0.0, exp_name):
    #   return

    optimizer = BayesianOptimization(f=func, pbounds=bounds, verbose=2, random_state=1)
    if verbose: print('BO opt...')
    utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.1e-1)
    # utility = UtilityFunction(kind="ucb", kappa=10, xi=0.1e-1)
    # utility = UtilityFunction(kind="poi", kappa=10, xi=1e-1)
    for i in range(bo_iterations):
        print("iteration: ", i)
        trials_so_far += 1
        try:
            next_point = optimizer.suggest(utility)
            next_point = list(next_point.values())
            target = func(next_point)
            result_logger.save_results(target, exp_name)
            logger.info("The input {} resulted in the the exception {}".format(input, target))
            target = validate_output(target, new_max)
            optimizer.register(params=next_point, target=target)
            if i % 10 == 0 and i is not 0:
                if len(optimizer.space.target) == 0:
                    utility.xi = utility.xi * 2
        except Exception as e:
            if isinstance(e, ValueError):
                num_fail += 1
                optimizer._space._target[-1] /= (10 ** num_fail)
            if verbose: print("Oops!", e.__class__, "occurred.")
            if verbose: print(e)
            # if verbose: logging.exception("Something awful happened!")
        finally:
            result_logger.update_runs_table(exp_name)
            continue

    if verbose: print(optimizer.max)
    val = optimizer.max['target']
    result_logger.save_results(val, exp_name)

def optimize(shared_lib: str, input_type: str, num_inputs: int, splitting: str, new_max: float):
    test_func.lib = shared_lib
    logger.info("Max value to replace: {}".format(str(new_max)))
    if input_type != "exp" or input_type != "fp":
        print('Invalid input type!')
        exit()

    assert num_inputs >= 1 and num_inputs <= 3

    funcs = ["max_inf", "min_inf", "max_under", "min_under"]
    num_inputs = [1, 2, 3, 4]
    exp_name = [shared_lib, input_type, splitting]
    logging.info('|'.join(exp_name))
    for f in funcs:
        initialize()
        g = partial(test_func.function_to_optimize, num_input=num_inputs, func_type=f, mode=input_type)
        for b in bounds(split=splitting, num_input=num_inputs, input_type=input_type):
            run_optimizer(b, g, new_max, '|'.join(exp_name))

# -------------- Results --------------
def print_results(shared_lib: str, number_sampling, range_splitting):
    result_logger.print_result(shared_lib, number_sampling, range_splitting, logger)

# -------------------------------------------------------
if __name__ == '__main__':
    optimize()
