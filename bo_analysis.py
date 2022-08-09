#!//usr/bin/env python3

import random_fp_generator
import logging
import math
import os
import numpy
import ctypes
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from functools import partial
from utils import *

#verbose = False
verbose = True
CUDA_LIB = ''
#MU = 1e-307
MU = 1.0
bo_iterations = 25 # number of iterations
smallest_subnormal = 4e-323
results = {}
runs_results = {}
trials_so_far = 0
trials_to_trigger = -1
trials_results = {}
random_results = {}
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

#----------------------------------------------------------------------------
# Ctype Wrappers
#----------------------------------------------------------------------------
def call_GPU_kernel_1(x):
  x0 = x[0]
  script_dir = os.path.abspath(os.path.dirname(__file__))
  lib_path = os.path.join(script_dir, CUDA_LIB)
  E = ctypes.cdll.LoadLibrary(lib_path)
  E.kernel_wrapper_1.restype = ctypes.c_double
  res = E.kernel_wrapper_1(ctypes.c_double(x0))
  return res

def call_GPU_kernel_2(x):
  x0,x1 = x[0], x[1]
  script_dir = os.path.abspath(os.path.dirname(__file__))
  lib_path = os.path.join(script_dir, CUDA_LIB)
  E = ctypes.cdll.LoadLibrary(lib_path)
  E.kernel_wrapper_1.restype = ctypes.c_double
  res = E.kernel_wrapper_1(ctypes.c_double(x0), ctypes.c_double(x1))
  return res

def call_GPU_kernel_3(x):
  x0,x1,x2= x[0], x[1], x[2]
  script_dir = os.path.abspath(os.path.dirname(__file__))
  lib_path = os.path.join(script_dir, CUDA_LIB)
  E = ctypes.cdll.LoadLibrary(lib_path)
  E.kernel_wrapper_1.restype = ctypes.c_double
  res = E.kernel_wrapper_1(ctypes.c_double(x0), ctypes.c_double(x1), ctypes.c_double(x2))
  return res

def call_GPU_kernel_4(x):
  x0,x1,x2,x3 = x[0], x[1], x[2], x[3]
  script_dir = os.path.abspath(os.path.dirname(__file__))
  lib_path = os.path.join(script_dir, CUDA_LIB)
  E = ctypes.cdll.LoadLibrary(lib_path)
  E.kernel_wrapper_1.restype = ctypes.c_double
  res = E.kernel_wrapper_1(ctypes.c_double(x0), ctypes.c_double(x1), ctypes.c_double(x2), ctypes.c_double(x3))
  return res

def function_to_optimize(x0, num_input, func_type = "max_inf", mode="fp"):
  if mode == "exp":
    x0 = numpy.power(10, x0)
  if num_input == 1:
    r = call_GPU_kernel_1(x0)
  elif num_input ==2:
    r = call_GPU_kernel_2(x0)
  elif num_input==3:
    r = call_GPU_kernel_3(x0)
  else:
    r = call_GPU_kernel_4(x0)

  if func_type == "min_inf":
    return -r
  elif func_type == "max_under":
    if r==0.0 or r==-0.0:
      return -MU
    elif r > -smallest_subnormal:
      return -r
  elif func_type == "min_under":
    if r==0.0 or r==-0.0:
      return MU
    elif r < smallest_subnormal:
      return -r
  return r   

#----------------------------------------------------------------------------
# Optimization loop
#----------------------------------------------------------------------------


# -------------- 4 Inputs ----------------------
#
# ----------------------------------------------

#----------------------------------------------------------------------------
# Results Checking
#----------------------------------------------------------------------------

def save_trials_to_trigger(exp_name: str):
  global trials_to_trigger, trials_so_far
  if trials_to_trigger == -1:
    trials_to_trigger = trials_so_far
    trials_results[exp_name] = trials_to_trigger

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

def save_results(val: float, exp_name: str):
  # Infinity
  if math.isinf(val):
    if exp_name not in results.keys():
      if val < 0.0:
        results[exp_name] = [1, 0, 0, 0, 0]
        save_trials_to_trigger(exp_name)
      else:
        results[exp_name] = [0, 1, 0, 0, 0]
        save_trials_to_trigger(exp_name)
    else:
      if val < 0.0:
        results[exp_name][0] += 1
      else:
        results[exp_name][1] += 1

  # Subnormals
  if numpy.isfinite(val):
    if val > -2.22e-308 and val < 2.22e-308:
      if val != 0.0 and val !=-0.0:
        if exp_name not in results.keys():
          if val < 0.0:
            results[exp_name] = [0, 0, 1, 0, 0]
            save_trials_to_trigger(exp_name)
          else:
            results[exp_name] = [0, 0, 0, 1, 0]
            save_trials_to_trigger(exp_name)
        else:
          if val < 0.0:
            results[exp_name][2] += 1
          else:
            results[exp_name][3] += 1

  if math.isnan(val):
    if exp_name not in results.keys():
      results[exp_name] = [0, 0, 0, 0, 1]
      save_trials_to_trigger(exp_name)
    else:
      results[exp_name][4] += 1

def update_runs_table(exp_name: str):
  if exp_name not in runs_results.keys():
    runs_results[exp_name] = 0
  else:
    runs_results[exp_name] += 1

def run_optimizer(bounds, func,new_max, exp_name):
  global trials_to_trigger, trials_so_far
  num_fail = 0
  trials_so_far = 0
  trials_to_trigger = -1
  if are_we_done(func, 0.0, exp_name):
    return
  optimizer = BayesianOptimization(f=func, pbounds=bounds, verbose=2, random_state=1)
  if verbose: print('BO opt...')
  utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.1e-1)
  #utility = UtilityFunction(kind="ucb", kappa=10, xi=0.1e-1)
  #utility = UtilityFunction(kind="poi", kappa=10, xi=1e-1)
  for i in range(bo_iterations):
    print("iteration: ", i)
    trials_so_far += 1
    next_point = 0.0
    try:
      next_point = optimizer.suggest(utility)
      target = func(**next_point)
      target = validate_output(next_point, target,new_max, exp_name)
      #if numpy.isnan(target):
        #next_point['x0'] += numpy.random.normal(0, .01)
        #target = func(**next_point)
      optimizer.register(params=next_point, target=target)
      if i%10 ==0 and i is not 0:
        if len(optimizer.space.target) == 0:
          utility.xi = utility.xi*2
    except Exception as e:
      if isinstance(e, ValueError):
        num_fail += 1
        optimizer._space._target[-1] /= (10**num_fail) 
      if verbose: print("Oops!", e.__class__, "occurred.")
      if verbose: print(e)
      #if verbose: logging.exception("Something awful happened!")
    finally:
      continue
     # target = func(**next_point)
     # try:
        #optimizer.register(params=next_point, target=target)
     # except KeyError:
      #  try:
       #   repeatedPointsProbed += 1
        #  print(
         #     f'Bayesian algorithm is attempting to probe an existing point: {next_point}.Continuing for now....')
          #if repeatedPointsProbed > 10:
           # print('The same point has been requested more than 10 times; quitting')
            #break
        #except AttributeError:
         # repeatedPointsProbed = 1
           
    update_runs_table(exp_name)

      # Check if we are done
      #if are_we_done(func, target, exp_name):
       # return
  if verbose: print(optimizer.max)
  val = optimizer.max['target']
  save_results(val, exp_name)

# input types: {"fp", "exp"}
def optimize(shared_lib: str, input_type: str, num_inputs: int, splitting: str, new_max:float):
  global CUDA_LIB
  global results
  global runs_results
  results = {}
  run_results = {}
  CUDA_LIB = shared_lib
  logger.info("Max value to replace: {}".format(str(new_max)))

  assert num_inputs >= 1 and num_inputs <= 3

  funcs = ["max_inf", "min_inf", "max_under", "min_under"]
  num_inputs = [1,2,3,4]
  
  for f in funcs:
    initialize()
    for b in bounds(split=splitting, num_input=num_inputs, input_type=input_type):
      exp_name = [shared_lib, input_type, splitting]
      g = partial(function_to_optimize(num_input=num_inputs, func_type = f, mode=input_type))
      logging.info('|'.join(exp_name))
      run_optimizer(b, g, new_max,  '|'.join(exp_name))

  else:
    print('Invalid input type!')
    exit()

#-------------- Results --------------
#lassen60_26904/cuda_code_acos.cu.so|fp|b_many :    [0, 0, 0, 0, 32]
#lassen60_26904/cuda_code_hypot.cu.so|fp|b_many :     [0, 0, 5, 0, 0]
def print_results(shared_lib: str, number_sampling, range_splitting):
  function_key = shared_lib+'|'+number_sampling+'|b_'+range_splitting
  fun_name = os.path.basename(shared_lib)
  print('-------------- Results --------------')
  print(fun_name)
  if len(results.keys()) > 0:
    total_exception = 0
    for key in results.keys():
      if function_key in key:
        print('Range:', key.split('|')[0])
        print('\tINF+:', results[key][0])
        print('\tINF-:', results[key][1])
        print('\tSUB-:', results[key][2])
        print('\tSUB-:', results[key][3])
        print('\tNaN :', results[key][4])
        #print('\tTotal Exception for range {}: {}'.format(key.split('|')[0], sum(results[key]))
        total_exception += sum(results[key])
    print('\tTotal Exception: ', total_exception)
    logger.info('\tTotal Exception: {} '.format(total_exception))
  else:
    print('\tINF+:', 0)
    print('\tINF-:', 0)
    print('\tSUB-:', 0)
    print('\tSUB-:', 0)
    print('\tNaN :', 0) 

  #print('\tRuns:', runs_results[key])
  #print('\n**** Runs ****')
  #for k in runs_results.keys():
  #  print(k, runs_results[k])

  print('')

# --------------- Random Sampling Optimizer -------------
def save_results_random(val: float, exp_name: str, unbounded: bool):
  found = False
  # Infinity
  if math.isinf(val):
    if exp_name not in random_results.keys():
      if val < 0.0:
        random_results[exp_name] = [1, 0, 0, 0, 0]
        found = True
      else:
        random_results[exp_name] = [0, 1, 0, 0, 0]
        found = True
    else:
      if val < 0.0:
        random_results[exp_name][0] += 1
        found = True
      else:
        random_results[exp_name][1] += 1
        found = True

  # Subnormals
  if numpy.isfinite(val):
    if val > -2.22e-308 and val < 2.22e-308:
      if val != 0.0 and val !=-0.0:
        if exp_name not in random_results.keys():
          if val < 0.0:
            random_results[exp_name] = [0, 0, 1, 0, 0]
            found = True
          else:
            random_results[exp_name] = [0, 0, 0, 1, 0]
            found = True
        else:
          if val < 0.0:
            random_results[exp_name][2] += 1
            found = True
          else:
            random_results[exp_name][3] += 1
            found = True

  if math.isnan(val):
    if exp_name not in random_results.keys():
      random_results[exp_name] = [0, 0, 0, 0, 1]
      found = True
    else:
      random_results[exp_name][4] += 1
      found = True

  if exp_name not in random_results.keys():
    random_results[exp_name] = [0,0,0,0,0]

  if unbounded:
    return False
  else:
    return found
  #return False

#_tmp_lassen593_3682/cuda_code_acos.cu.so|RANDOM :    [0, 0, 0, 0, 271]
def print_results_random(shared_lib):
  key = shared_lib+'|RANDOM'
  fun_name = os.path.basename(shared_lib)
  print('-------------- Results --------------')
  print(fun_name)
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

# Calls to wrappers:
# call_GPU_kernel_1(x0)
# call_GPU_kernel_1(x0,x1)
# call_GPU_kernel_1(x0,x1,x2)
def optimize_randomly(shared_lib: str, num_inputs: int, max_iters: int, unbounded: bool):
  global CUDA_LIB
  CUDA_LIB = shared_lib
  exp_name = shared_lib+'|'+'RANDOM'
  for i in range(max_iters):
    if num_inputs == 1:
      x0 = random_fp_generator.fp64_generate_number()
      r = call_GPU_kernel_1(x0)
      found = save_results_random(r, exp_name, unbounded)
      if found: break
    elif num_inputs == 2:
      x0 = random_fp_generator.fp64_generate_number()
      x1 = random_fp_generator.fp64_generate_number()
      r = call_GPU_kernel_2(x0,x1)
      found = save_results_random(r, exp_name, unbounded)
      if found: break
    elif num_inputs == 3:
      x0 = random_fp_generator.fp64_generate_number()
      x1 = random_fp_generator.fp64_generate_number()
      x2 = random_fp_generator.fp64_generate_number()
      r = call_GPU_kernel_3(x0,x1,x2)
      found = save_results_random(r, exp_name, unbounded)
      if found: break

def validate_output(input, output, new_max,  exp_name):
  if numpy.isposinf(output):
    save_results(output, exp_name)
    logger.info("The input {} resulted in the the exception {}".format(input, output))
    #output = 8.95e+305
    output = new_max
  elif numpy.isneginf(output):
    save_results(output, exp_name)
    logger.info("The input {} resulted in the the exception {}".format(input, output))
    output = -new_max
  elif numpy.isnan(output):
    save_results(output, exp_name)
    logger.info("The input {} resulted in the the exception {}".format(input, output))
    output = new_max
  return output

# -------------------------------------------------------

if __name__ == '__main__':
  optimize()

