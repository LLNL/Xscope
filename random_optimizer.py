import random_fp_generator
from test_function import *
from utils import *
# --------------- Random Sampling Optimizer -------------

test_func = TestFunction(CUDA_LIB)
result_logger = ResultLogger()

def save_results_random(val: float, exp_name: str, unbounded: bool):
    found = False
    # Infinity
    if math.isinf(val):
        if exp_name not in result_logger.random_results.keys():
            if val < 0.0:
                result_logger.random_results[exp_name] = [1, 0, 0, 0, 0]
                found = True
            else:
                result_logger.random_results[exp_name] = [0, 1, 0, 0, 0]
                found = True
        else:
            if val < 0.0:
                result_logger.random_results[exp_name][0] += 1
                found = True
            else:
                result_logger.random_results[exp_name][1] += 1
                found = True

    # Subnormals
    if numpy.isfinite(val):
        if val > -2.22e-308 and val < 2.22e-308:
            if val != 0.0 and val != -0.0:
                if exp_name not in result_logger.random_results.keys():
                    if val < 0.0:
                        result_logger.random_results[exp_name] = [0, 0, 1, 0, 0]
                        found = True
                    else:
                        result_logger.random_results[exp_name] = [0, 0, 0, 1, 0]
                        found = True
                else:
                    if val < 0.0:
                        result_logger.random_results[exp_name][2] += 1
                        found = True
                    else:
                        result_logger.random_results[exp_name][3] += 1
                        found = True

    if math.isnan(val):
        if exp_name not in result_logger.random_results.keys():
            result_logger.random_results[exp_name] = [0, 0, 0, 0, 1]
            found = True
        else:
            result_logger.random_results[exp_name][4] += 1
            found = True

    if exp_name not in result_logger.random_results.keys():
        result_logger.random_results[exp_name] = [0, 0, 0, 0, 0]

    if unbounded:
        return False
    else:
        return found
    # return False


# _tmp_lassen593_3682/cuda_code_acos.cu.so|RANDOM :    [0, 0, 0, 0, 271]
def print_results_random(shared_lib):
    key = shared_lib + '|RANDOM'
    fun_name = os.path.basename(shared_lib)
    print('-------------- Results --------------')
    print(fun_name)
    random_results = result_logger.get_random_results()
    if key in random_results.keys():
        print('\tINF+:', random_results[key][0])
        print('\tINF-:', random_results[key][1])
        print('\tSUB-:', random_results[key][2])
        print('\tSUB-:', random_results[key][3])
        print('\tNaN :', random_results[key][4])
    else:
        print('\tINF+:', 0)
        print('\tINF-:', 0)
        print('\tSUB-:', 0)
        print('\tSUB-:', 0)
        print('\tNaN :', 0)
    print('')

def optimize_randomly(shared_lib: str, num_inputs: int, max_iters: int, unbounded: bool):
    global CUDA_LIB
    CUDA_LIB = shared_lib
    exp_name = shared_lib + '|' + 'RANDOM'
    for i in range(max_iters):
        if num_inputs == 1:
            x0 = random_fp_generator.fp64_generate_number()
            r = test_func.call_GPU_kernel_1(x0)
            found = save_results_random(r, exp_name, unbounded)
            if found: break
        elif num_inputs == 2:
            x0 = random_fp_generator.fp64_generate_number()
            x1 = random_fp_generator.fp64_generate_number()
            r = test_func.call_GPU_kernel_2([x0, x1])
            found = save_results_random(r, exp_name, unbounded)
            if found: break
        elif num_inputs == 3:
            x0 = random_fp_generator.fp64_generate_number()
            x1 = random_fp_generator.fp64_generate_number()
            x2 = random_fp_generator.fp64_generate_number()
            r = test_func.call_GPU_kernel_3([x0, x1, x2])
            found = save_results_random(r, exp_name, unbounded)
            if found: break