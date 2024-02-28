from global_init import *
from utils.utils import ResultLogger
from test_function import TestFunction
from utils.input_bounds import Input_bound
import torch
from BO.BaysianOptimization import *
from BO.MultiObjBO import *
import numpy as np
import traceback
# from old_xscope import *

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

def optimize(shared_lib: str, input_type: str, num_inputs: int, splitting: int, inputs_type="single"):
    result_logger = ResultLogger()
    input_ranges = np.array([-max_normal,max_normal], dtype=np.double)
    is_custom_func = False
    if inputs_type == "single":
        is_input_array = False
    else:
        is_input_array = True
    num_task = 1
    test_func = TestFunction(num_input=num_inputs, mode=input_type, input_ranges=input_ranges, num_task=num_task, is_custom_func=is_custom_func, test_device= device, is_input_array=is_input_array)
    if not is_custom_func:
        test_func.set_kernel(shared_lib)
    # logger.info("Max value to replace: {}"v  .format(str(new_max)))
    if input_type != "exp" and input_type != "fp":
        print('Invalid input type!')
        exit()

    exp_name = [shared_lib, input_type, str(splitting)]
    exp_name ='|'.join(exp_name)
    logging.info(exp_name)
    print("experience name: ", exp_name)

    # bgrt_bo_compare = "bgrt_bo_compare.csv"

    funcs = ["max_inf", "min_inf", "max_under", "min_under"]
    # funcs = ["max_inf", "min_under"]

    result_logger.start_time()
    BO_bounds = Input_bound(num_input=num_inputs, input_type=input_type, input_range=input_ranges, params_per_group=num_inputs)
    print("Begin")
    BO_bounds.generate_bounds(splitting)
    for f in funcs:
        start_time_one_f = time.time()
        test_func.set_fn_type(f)
        num_run = 0
        if BO_bounds.ignore_params is None:
            bound_done = 0
            for bounds in BO_bounds.bounds_set:
                start_one_bound = time.time()
                BO_bounds.update_dim(bounds)
                train_start = time.time()
                if num_task == 1:
                    bo = BaysianOptimization(test_func, bounds=BO_bounds)
                else:
                    bo = MultiObjectiveBO(test_func, num_task=num_task, bounds=BO_bounds)
                try:
                    bo.train()
                except:
                    traceback.print_exc()
                # bo.memory_measure()
                # print(bo.train_y)
                result_logger.log_result(bo.results)
                bo.result_file.close()
                del bo
                torch.cuda.empty_cache()
                # max_mem = torch.cuda.max_memory_allocated(device=None)
                # print(f"After training, gpu used {round(max_mem / (1024 ** 3), 2)} memory") 
                bound_done+=1
                # print("train_time: ", time.time()-train_start)
                # print("number of bound finished: {}".format(bound_done))
                print("time to run one bound: ", time.time()- start_one_bound)
        else:
            # bounds_combination = []
            # early exploration
            for ignore_param in BO_bounds.ignore_params:
                BO_bounds.generate_bounds(splitting, ignore_param)
                print("Bounds shape: ", BO_bounds.bounds.shape)
                test_func.set_ignore_params(ignore_params=ignore_param)
                print("number of bound set: ", len(BO_bounds.bounds_set))
                print("number of params group: ", len(BO_bounds.ignore_params))
                bound_done = 0
                start_one_bound = time.time()
                for bounds in BO_bounds.bounds_set:
                    start_one_bound = time.time()
                    print("number of bound finished: {}".format(bound_done))
                    try:
                        BO_bounds.update_dim(bounds)
                        if num_task == 1:
                            bo = BaysianOptimization(test_func, bounds=BO_bounds, exp_name=exp_name)
                        else:
                            bo = MultiObjectiveBO(test_func, num_task=num_task, bounds=BO_bounds, exp_name=exp_name)
                        bo.train()
                        bo.memory_measure()
                        result_logger.log_result(bo.results)
                        # bounds_combination.append(bo.best_bound)
                        bo.result_file.close()
                        del bo.train_y
                        del bo.train_x
                        del bo
                        torch.cuda.empty_cache()
                        num_run+=1
                        bound_done+=1
                        print("time to run one bound: ", time.time()- start_one_bound)
                    except Exception:
                        print(traceback.format_exc())
                        continue
        # print("time for one function: ", time.time()-start_time_one_f)        
        #         # # thorough exploration
        #         # bounds_combination = torch.cat(bounds_combination, dim=-1).unsqueeze(0)
        #         # print(bounds_combination)
        #         # BO_bounds.update_bound(bounds_combination)
        #         # test_func.set_ignore_params([])
        #         # combine_bo = BaysianOptimization(test_func, bounds=BO_bounds)
        #         # combine_bo.train()
        #         # result_logger.log_result(combine_bo.results)
        #         # print(combine_bo.results)
        #         # print("Execution time per function: ", time.time()-start_time)
        #         # del combine_bo

    result_logger.log_time()
    max_mem = torch.cuda.max_memory_allocated(device=None)
    print(f"After training, gpu used {round(max_mem / (1024 ** 3), 2)} memory") 
    print("Total execution time: ", result_logger.execution_time)
    print(result_logger.results)
    logging.info("Result Summary: {}".format(result_logger.results))
    logging.info("Total execution time: {}".format(result_logger.execution_time))
    # result_logger.log_time()
    # print("execution time: ", result_logger.execution_time)
    # print(result_logger.results)
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
