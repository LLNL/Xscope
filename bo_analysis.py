from global_init import *
from utils.utils import ResultLogger
from test_function import TestFunction
from utils.input_bounds import Input_bound
import torch
from BO.BaysianOptimization import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
verbose = True
CUDA_LIB = ''
bo_iterations = 25  # number of iteration

logging.basicConfig(filename='Xscope.log', level=logging.INFO)
logger = logging.getLogger(__name__)
max_normal = 1e+307
def set_max_iterations(n: int):
    global bo_iterations
    bo_iterations = n

def optimize(shared_lib: str, input_type: str, num_inputs: int, splitting: int):
    result_logger = ResultLogger()
    test_func = TestFunction(num_input=num_inputs, mode=input_type)
    test_func.set_kernel(shared_lib)
    # logger.info("Max value to replace: {}".format(str(new_max)))
    if input_type != "exp" and input_type != "fp":
        print('Invalid input type!')
        exit()

    exp_name = [shared_lib, input_type, str(splitting)]
    logging.info('|'.join(exp_name))

    # bgrt_bo_compare = "bgrt_bo_compare.csv"

    # funcs = ["max_inf", "min_inf", "max_under", "min_under"]
    funcs = ["max_under", "min_under"]

    result_logger.start_time()
    for f in funcs:
        test_func.set_fn_type(f)
        BO_bounds = Input_bound(split=splitting, num_input=num_inputs, input_type=input_type, input_range=[-1e-100, 0.0], f_type=f)
        padded_y =  torch.as_tensor(test_func.eval(BO_bounds.padded_x), device=device, dtype=dtype)
        BO_bounds.set_padded_y(padded_y)
        bo = BaysianOptimization(test_func, bounds=BO_bounds)
        bo.train()
        result_logger.log_result(bo.results)
        print(bo.train_y)
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
    del test_func


# -------------------------------------------------------
if __name__ == '__main__':
    optimize()
