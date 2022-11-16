#!//usr/bin/env python3
from functools import partial
from test_function import *
from BaysianOptimization import *
from BO_JAX import *
from utils import Input_bound

# verbose = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
verbose = True
CUDA_LIB = ''
bo_iterations = 25  # number of iteration
logging.basicConfig(filename='Xscope.log', level=logging.INFO)
logger = logging.getLogger(__name__)

def set_max_iterations(n: int):
    global bo_iterations
    bo_iterations = n

test_func = TestFunction()


def optimize(shared_lib: str, input_type: str, num_inputs: int, splitting: str, new_max: float):
    result_logger = ResultLogger()

    test_func.set_kernel(shared_lib)
    logger.info("Max value to replace: {}".format(str(new_max)))
    if input_type != "exp" and input_type != "fp":
        print('Invalid input type!')
        exit()

    assert num_inputs >= 1 and num_inputs <= 3


    exp_name = [shared_lib, input_type, splitting]
    logging.info('|'.join(exp_name))

    # bgrt_bo_compare = "bgrt_bo_compare.csv"

    funcs = ["max_inf", "min_inf", "max_under", "min_under"]

    result_logger.start_time()
    for f in funcs:
        g = partial(test_func.function_to_optimize, num_input=num_inputs, func_type=f, mode=input_type)
        BO_bounds = Input_bound(split=splitting, num_input=num_inputs, input_type=input_type)
        bo = BaysianOptimization(g, bounds=BO_bounds)
        bo.train()
        result_logger.log_result(bo.results)
        del bo
    result_logger.log_time()
    print("execution time: ", result_logger.execution_time)
    print(result_logger.results)
    # bgrt_bo_df, bgrt_interval_df = result_logger.summarize_result(shared_lib)
    #
    # if num_inputs==1:
    #     bgrt_interval_density = "bgrt_interval_density_1.csv"
    # elif num_inputs==2:
    #     bgrt_interval_density = "bgrt_interval_density_2.csv"
    # else:
    #     bgrt_interval_density = "bgrt_interval_density_3.csv"
    #
    # result_logger.write_result_to_file(bgrt_bo_compare, bgrt_bo_df)
    # result_logger.write_result_to_file(bgrt_interval_density, bgrt_interval_df)

    del result_logger

# -------------------------------------------------------
if __name__ == '__main__':
    optimize()
