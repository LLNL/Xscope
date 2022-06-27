#!//usr/bin/env python3

import random_fp_generator
import logging
import math
import os
import numpy
import ctypes
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

verbose = False
#verbose = True
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
def call_GPU_kernel_1(x0):
  script_dir = os.path.abspath(os.path.dirname(__file__))
  lib_path = os.path.join(script_dir, CUDA_LIB)
  E = ctypes.cdll.LoadLibrary(lib_path)
  E.kernel_wrapper_1.restype = ctypes.c_double
  res = E.kernel_wrapper_1(ctypes.c_double(x0))
  return res

def call_GPU_kernel_2(x0, x1):
  script_dir = os.path.abspath(os.path.dirname(__file__))
  lib_path = os.path.join(script_dir, CUDA_LIB)
  E = ctypes.cdll.LoadLibrary(lib_path)
  E.kernel_wrapper_1.restype = ctypes.c_double
  res = E.kernel_wrapper_1(ctypes.c_double(x0), ctypes.c_double(x1))
  return res

def call_GPU_kernel_3(x0, x1, x2):
  script_dir = os.path.abspath(os.path.dirname(__file__))
  lib_path = os.path.join(script_dir, CUDA_LIB)
  E = ctypes.cdll.LoadLibrary(lib_path)
  E.kernel_wrapper_1.restype = ctypes.c_double
  res = E.kernel_wrapper_1(ctypes.c_double(x0), ctypes.c_double(x1), ctypes.c_double(x2))
  return res

def call_GPU_kernel_4(x0, x1, x2, x3):
  script_dir = os.path.abspath(os.path.dirname(__file__))
  lib_path = os.path.join(script_dir, CUDA_LIB)
  E = ctypes.cdll.LoadLibrary(lib_path)
  E.kernel_wrapper_1.restype = ctypes.c_double
  res = E.kernel_wrapper_1(ctypes.c_double(x0), ctypes.c_double(x1), ctypes.c_double(x2), ctypes.c_double(x3))
  return res


#----------------------------------------------------------------------------
# Black box functions
#----------------------------------------------------------------------------

#def black_box_function(x0, x1, x2):
#  x0_fp = 1.0 * math.pow(10, x0)
#  x1_fp = 1.0 * math.pow(10, x1)
#  x2_fp = 1.0 * math.pow(10, x2)

  #return -call_GPU_kernel(x0, x1, x2)
  #r = call_GPU_kernel(x0, x1, x2)
#  r = call_GPU_kernel(x0_fp, x1_fp, x2_fp)
#  smallest_subnormal = -4.94e-323
#  if r==0.0 or r==-0.0:
#    return -1.0;
#  elif r > smallest_subnormal:
#    return -r
#  return r

#****************************** 1 Input ***********************************
# --------- Based on FP inputs --------
# Function goals: (1) maximize (2) find INF (3) use fp inputs
def function_max_inf_fp_1(x0):
  return call_GPU_kernel_1(x0)

# Function goals: (1) minimize (2) find INF (3) use fp inputs
def function_min_inf_fp_1(x0):
  return -call_GPU_kernel_1(x0)

# Function goals: (1) maximize (2) find Underflows (3) use fp inputs
def function_max_under_fp_1(x0):
  r = call_GPU_kernel_1(x0)
  if r==0.0 or r==-0.0:
    return -MU
  elif r > -smallest_subnormal:
    return -r
  return r

# Function goals: (1) minimize (2) find Underflows (3) use fp inputs
def function_min_under_fp_1(x0):
  r = call_GPU_kernel_1(x0)
  if r==0.0 or r==-0.0:
    return MU
  elif r < smallest_subnormal:
    return -r
  return r

# --------- Based on Exponent inputs --------
# Function goals: (1) maximize (2) find INF (3) use exp inputs
def function_max_inf_exp_1(x0):
  x0_fp = 1.0 * math.pow(10, x0)
  return call_GPU_kernel_1(x0_fp)

# Function goals: (1) minimize (2) find INF (3) use exp inputs
def function_min_inf_exp_1(x0):
  x0_fp = 1.0 * math.pow(10, x0)
  return -call_GPU_kernel_1(x0_fp)

# Function goals: (1) maximize (2) find Underflows (3) use exp inputs
def function_max_under_exp_1(x0):
  x0_fp = 1.0 * math.pow(10, x0)
  r = call_GPU_kernel_1(x0_fp)
  if r==0.0 or r==-0.0:
    return -MU
  elif r > -smallest_subnormal:
    return -r
  return r

# Function goals: (1) minimize (2) find Underflows (3) use exp inputs
def function_min_under_exp_1(x0):
  x0_fp = 1.0 * math.pow(10, x0)
  r = call_GPU_kernel_1(x0_fp)
  if r==0.0 or r==-0.0:
    return MU
  elif r < smallest_subnormal:
    return -r
  return r


#****************************** 2 Inputs ***********************************
# --------- Based on FP inputs --------
# Function goals: (1) maximize (2) find INF (3) use fp inputs
def function_max_inf_fp_2(x0, x1):
  return call_GPU_kernel_2(x0, x1)

# Function goals: (1) minimize (2) find INF (3) use fp inputs
def function_min_inf_fp_2(x0, x1):
  return -call_GPU_kernel_2(x0, x1)

# Function goals: (1) maximize (2) find Underflows (3) use fp inputs
def function_max_under_fp_2(x0, x1):
  r = call_GPU_kernel_2(x0, x1)
  if r==0.0 or r==-0.0:
    return -MU
  elif r > -smallest_subnormal:
    return -r
  return r

# Function goals: (1) minimize (2) find Underflows (3) use fp inputs
def function_min_under_fp_2(x0, x1):
  r = -call_GPU_kernel_2(x0, x1)
  if r==0.0 or r==-0.0:
    return MU
  elif r > smallest_subnormal:
    return -r
  return r

# --------- Based on Exponent inputs --------
# Function goals: (1) maximize (2) find INF (3) use exp inputs
def function_max_inf_exp_2(x0, x1):
  x0_fp = 1.0 * math.pow(10, x0)
  x1_fp = 1.0 * math.pow(10, x1)
  return call_GPU_kernel_2(x0_fp, x1_fp)

# Function goals: (1) minimize (2) find INF (3) use exp inputs
def function_min_inf_exp_2(x0, x1):
  x0_fp = 1.0 * math.pow(10, x0)
  x1_fp = 1.0 * math.pow(10, x1)
  return -call_GPU_kernel_2(x0_fp, x1_fp)

# Function goals: (1) maximize (2) find Underflows (3) use exp inputs
def function_max_under_exp_2(x0, x1):
  x0_fp = 1.0 * math.pow(10, x0)
  x1_fp = 1.0 * math.pow(10, x1)
  r = call_GPU_kernel_2(x0_fp, x1_fp)
  if r==0.0 or r==-0.0:
    return -MU
  elif r > -smallest_subnormal:
    return -r
  return r

# Function goals: (1) minimize (2) find Underflows (3) use exp inputs
def function_min_under_exp_2(x0, x1):
  x0_fp = 1.0 * math.pow(10, x0)
  x1_fp = 1.0 * math.pow(10, x1)
  r = -call_GPU_kernel_2(x0_fp, x1_fp)
  if r==0.0 or r==-0.0:
    return MU
  elif r > smallest_subnormal:
    return -r
  return r


#****************************** 3 Inputs ***********************************
# --------- Based on FP inputs --------
# Function goals: (1) maximize (2) find INF (3) use fp inputs
def function_max_inf_fp_3(x0, x1, x2):
  return call_GPU_kernel_3(x0, x1, x2)

# Function goals: (1) minimize (2) find INF (3) use fp inputs
def function_min_inf_fp_3(x0, x1, x2):
  return -call_GPU_kernel_3(x0, x1, x2)

# Function goals: (1) maximize (2) find Underflows (3) use fp inputs
def function_max_under_fp_3(x0, x1, x2):
  r = call_GPU_kernel_3(x0, x1, x2)
  if r==0.0 or r==-0.0:
    return -MU
  elif r > -smallest_subnormal:
    return -r
  return r

# Function goals: (1) minimize (2) find Underflows (3) use fp inputs
def function_min_under_fp_3(x0, x1, x2):
  r = call_GPU_kernel_3(x0, x1, x2)
  if r==0.0 or r==-0.0:
    return MU
  elif r < smallest_subnormal:
    return -r
  return r

# --------- Based on Exponent inputs --------
# Function goals: (1) maximize (2) find INF (3) use exp inputs
def function_max_inf_exp_3(x0, x1, x2):
  x0_fp = 1.0 * math.pow(10, x0)
  x1_fp = 1.0 * math.pow(10, x1)
  x2_fp = 1.0 * math.pow(10, x2)
  return call_GPU_kernel_3(x0_fp, x1_fp, x2_fp)

# Function goals: (1) minimize (2) find INF (3) use exp inputs
def function_min_inf_exp_3(x0, x1, x2):
  x0_fp = 1.0 * math.pow(10, x0)
  x1_fp = 1.0 * math.pow(10, x1)
  x2_fp = 1.0 * math.pow(10, x2)
  return -call_GPU_kernel_3(x0_fp, x1_fp, x2_fp)

# Function goals: (1) maximize (2) find Underflows (3) use exp inputs
def function_max_under_exp_3(x0, x1, x2):
  x0_fp = 1.0 * math.pow(10, x0)
  x1_fp = 1.0 * math.pow(10, x1)
  x2_fp = 1.0 * math.pow(10, x2)
  r = call_GPU_kernel_3(x0_fp, x1_fp, x2_fp)
  if r==0.0 or r==-0.0:
    return -MU
  elif r > -smallest_subnormal:
    return -r
  return r

# Function goals: (1) minimize (2) find Underflows (3) use exp inputs
def function_min_under_exp_3(x0, x1, x2):
  x0_fp = 1.0 * math.pow(10, x0)
  x1_fp = 1.0 * math.pow(10, x1)
  x2_fp = 1.0 * math.pow(10, x2)
  r = call_GPU_kernel_3(x0_fp, x1_fp, x2_fp)
  if r==0.0 or r==-0.0:
    return MU
  elif r < smallest_subnormal:
    return -r
  return r

#****************************** 4 Inputs ***********************************
# --------- Based on FP inputs --------
# Function goals: (1) maximize (2) find INF (3) use fp inputs
def function_max_inf_fp_4(x0, x1, x2, x3):
  return call_GPU_kernel_4(x0, x1, x2, x3)

# Function goals: (1) minimize (2) find INF (3) use fp inputs
def function_min_inf_fp_4(x0, x1, x2, x3):
  return -call_GPU_kernel_4(x0, x1, x2, x3)

# Function goals: (1) maximize (2) find Underflows (3) use fp inputs
def function_max_under_fp_4(x0, x1, x2, x3):
  r = call_GPU_kernel_4(x0, x1, x2, x3)
  if r==0.0 or r==-0.0:
    return -MU
  elif r > -smallest_subnormal:
    return -r
  return r

# Function goals: (1) minimize (2) find Underflows (3) use fp inputs
def function_min_under_fp_4(x0, x1, x2, x3):
  r = call_GPU_kernel_4(x0, x1, x2, x3)
  if r==0.0 or r==-0.0:
    return MU
  elif r < smallest_subnormal:
    return -r
  return r

# --------- Based on Exponent inputs --------
# Function goals: (1) maximize (2) find INF (3) use exp inputs
def function_max_inf_exp_4(x0, x1, x2, x3):
  x0_fp = 1.0 * math.pow(10, x0)
  x1_fp = 1.0 * math.pow(10, x1)
  x2_fp = 1.0 * math.pow(10, x2)
  x3_fp = 1.0 * math.pow(10, x3)
  return call_GPU_kernel_4(x0_fp, x1_fp, x2_fp, x3_fp)

# Function goals: (1) minimize (2) find INF (3) use exp inputs
def function_min_inf_exp_4(x0, x1, x2, x3):
  x0_fp = 1.0 * math.pow(10, x0)
  x1_fp = 1.0 * math.pow(10, x1)
  x2_fp = 1.0 * math.pow(10, x2)
  x3_fp = 1.0 * math.pow(10, x3)
  return -call_GPU_kernel_4(x0_fp, x1_fp, x2_fp, x3_fp)

# Function goals: (1) maximize (2) find Underflows (3) use exp inputs
def function_max_under_exp_3(x0, x1, x2, x3):
  x0_fp = 1.0 * math.pow(10, x0)
  x1_fp = 1.0 * math.pow(10, x1)
  x2_fp = 1.0 * math.pow(10, x2)
  x3_fp = 1.0 * math.pow(10, x3)
  r = call_GPU_kernel_4(x0_fp, x1_fp, x2_fp, x3_fp)
  if r==0.0 or r==-0.0:
    return -MU
  elif r > -smallest_subnormal:
    return -r
  return r

# Function goals: (1) minimize (2) find Underflows (3) use exp inputs
def function_min_under_exp_3(x0, x1, x2, x3):
  x0_fp = 1.0 * math.pow(10, x0)
  x1_fp = 1.0 * math.pow(10, x1)
  x2_fp = 1.0 * math.pow(10, x2)
  x3_fp = 1.0 * math.pow(10, x3)
  r = call_GPU_kernel_4(x0_fp, x1_fp, x2_fp, x3_fp)
  if r==0.0 or r==-0.0:
    return MU
  elif r < smallest_subnormal:
    return -r
  return r

#----------------------------------------------------------------------------
# Optimization loop
#----------------------------------------------------------------------------
max_normal = 1e+307

# -------------- 1 Input ----------------------
def bounds_fp_whole_1():
  b = []
  b.append({'x0': (-max_normal, max_normal)})
  return b

def bounds_fp_two_1():
  b = []
  b.append({'x0': (-max_normal, 0)})
  b.append({'x0': (0, max_normal)})
  return b

def bounds_fp_many_1():
  b = []
  limits = [0.0, 1e-307, 1e-100, 1e-10, 1e-1, 1e0, 1e+1, 1e+10, 1e+100, 1e+307]
  ranges = []
  for i in range(len(limits)-1):
    x = limits[i]
    y = limits[i+1]
    t = (min(x,y), max(x,y))
    ranges.append(t)
    x = -limits[i]
    y = -limits[i+1]
    t = (min(x,y), max(x,y))
    ranges.append(t)
  
  for r1 in ranges:
    b.append({'x0': r1})

  return b
  
def bounds_exp_whole_1():
  b = []
  b.append({'x0': (-307, 307)})
  return b

def bounds_exp_two_1():
  b = []
  b.append({'x0': (-307, 0)})
  b.append({'x0': (0, 307)})
  return b

def bounds_exp_many_1():
  b = []
  limits = [-307, -100, -10, -1, 0, +1, +10, +100, +307]
  for i in range(len(limits)-1):
    x = limits[i]
    y = limits[i+1]
    t = (min(x,y), max(x,y))
    b.append({'x0': t}) 
 
  return b

# -------------- 2 Inputs ----------------------
def bounds_fp_whole_2():
  b = []
  b.append({'x0': (-max_normal, max_normal), 'x1': (-max_normal, max_normal)})
  return b

def bounds_fp_two_2():
  b = []
  b.append({'x0': (-max_normal, 0), 'x1': (-max_normal, 0)})
  b.append({'x0': (0, max_normal), 'x1': (0, max_normal)})
  return b

def bounds_fp_many_2():
  b = []
  # {'target': -5e-324, 'params': {'x0': 1e-100, 'x1': 3.234942383692966}}
  #b.append({'x0': (1e-100, 1e-300), 'x1': (0, 4.0)}) # finds subnormal in pow(x0, x1)
  #b.append({'x0': (1e-100, 1e-307), 'x1': (0, 1e+1)}) # finds subnormal in pow(x0, x1)

  # finds subnormal in pow(x0, x1): 
  # {'target': -2.9950764292998e-310, 'params': {'x0': 6.639021130307829e-101, 'x1': 3.089739399676855}}
  # MU = 1.0
  # bo_iterations = 25 # number of iterations
  # utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.1e-1)
  #b.append({'x0': (1e-307, 1e-100), 'x1': (1e0, 1e+1)}) 

  limits = [0.0, 1e-307, 1e-100, 1e-10, 1e-1, 1e0, 1e+1, 1e+10, 1e+100, 1e+307]
  ranges = []
  for i in range(len(limits)-1):
    x = limits[i]
    y = limits[i+1]
    t = (min(x,y), max(x,y))
    ranges.append(t)
    x = -limits[i]
    y = -limits[i+1]
    t = (min(x,y), max(x,y))
    ranges.append(t)
  
  for r1 in ranges:
    for r2 in ranges:
      b.append({'x0': r1, 'x1': r2})

  return b

def bounds_exp_whole_2():
  b = []
  b.append({'x0': (-307, 307), 'x1': (-307, 307)})
  return b

def bounds_exp_two_2():
  b = []
  b.append({'x0': (-307, 0), 'x1': (-307, 0)})
  b.append({'x0': (0, 307),'x1': (0, 307)})
  return b

def bounds_exp_many_2():
  b = []
  limits = [-307, -100, -10, -1, 0, +1, +10, +100, +307]
  ranges = []
  for i in range(len(limits)-1):
    x = limits[i]
    y = limits[i+1]
    t = (min(x,y), max(x,y))
    ranges.append(t)
  
  for r1 in ranges:
    for r2 in ranges:
      b.append({'x0': r1, 'x1': r2})

  return b

# -------------- 3 Inputs ----------------------
def bounds_fp_whole_3():
  b = []
  b.append({'x0': (-max_normal, max_normal), 'x1': (-max_normal, max_normal), 'x2': (-max_normal, max_normal)})
  return b

def bounds_fp_two_3():
  b = []
  b.append({'x0': (-max_normal, 0), 'x1': (-max_normal, 0), 'x2': (-max_normal, 0)})
  b.append({'x0': (0, max_normal), 'x1': (0, max_normal), 'x2': (0, max_normal)})
  return b

def bounds_fp_many_3():
  b = []
  limits = [0.0, 1e-307, 1e-100, 1e-10, 1e-1, 1e0, 1e+1, 1e+10, 1e+100, 1e+307]
  ranges = []
  for i in range(len(limits)-1):
    x = limits[i]
    y = limits[i+1]
    t = (min(x,y), max(x,y))
    ranges.append(t)
    x = -limits[i]
    y = -limits[i+1]
    t = (min(x,y), max(x,y))
    ranges.append(t)
  
  for r1 in ranges:
    for r2 in ranges:
      #for r3 in ranges:
      b.append({'x0': r1, 'x1': r2, 'x2': r2})

  return b

def bounds_exp_whole_3():
  b = []
  b.append({'x0': (-307, 307), 'x1': (-307, 307), 'x2': (-307, 307)})
  return b

def bounds_exp_two_3():
  b = []
  b.append({'x0': (-307, 0), 'x1': (-307, 0), 'x2': (-307, 0)})
  b.append({'x0': (0, 307),'x1': (0, 307), 'x3': (0, 307)})
  return b

def bounds_exp_many_3():
  b = []
  limits = [-307, -100, -10, -1, 0, +1, +10, +100, +307]
  ranges = []
  for i in range(len(limits)-1):
    x = limits[i]
    y = limits[i+1]
    t = (min(x,y), max(x,y))
    ranges.append(t)
  
  for r1 in ranges:
    for r2 in ranges:
      #for r3 in ranges:
      b.append({'x0': r1, 'x1': r2, 'x2': r2})

  return b

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

def are_we_done(func, recent_val, exp_name):
  global found_inf_pos, found_inf_neg, found_under_pos, found_under_neg

  # Finding INF+
  if 'max_inf' in func.__name__:
    if found_inf_pos:
      return True
    else:
      if is_inf_pos(recent_val):
        found_inf_pos = True
        save_results(recent_val, exp_name)
        return True

  # Finding INF-
  elif 'min_inf' in func.__name__:
    if found_inf_neg:
      return True
    else:
      if is_inf_neg(recent_val):
        found_inf_neg = True
        save_results(recent_val, exp_name)
        return True

  # Finding Under-
  elif 'max_under' in func.__name__:
    if found_under_neg:
      return True
    else:
      if is_under_neg(recent_val):
        found_under_neg = True
        save_results(recent_val, exp_name)
        return True

  # Finding Under+
  elif 'min_under' in func.__name__:
    if found_under_pos:
      return True
    else:
      if is_under_pos(recent_val):
        found_under_pos = True
        save_results(recent_val, exp_name)
        return True

  return False

def update_runs_table(exp_name: str):
  if exp_name not in runs_results.keys():
    runs_results[exp_name] = 0
  else:
    runs_results[exp_name] += 1

def run_optimizer(bounds, func, exp_name):
  global trials_to_trigger, trials_so_far
  trials_so_far = 0
  trials_to_trigger = -1
  if are_we_done(func, 0.0, exp_name):
    return
  optimizer = BayesianOptimization(f=func, pbounds=bounds, verbose=2, random_state=1)
  try:
    if verbose: print('BO opt...')
    utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.1e-1)
    #utility = UtilityFunction(kind="ucb", kappa=10, xi=0.1e-1)
    #utility = UtilityFunction(kind="poi", kappa=10, xi=1e-1)
    for _ in range(bo_iterations):
      trials_so_far += 1
      next_point = optimizer.suggest(utility)
      target = func(**next_point)
      optimizer.register(params=next_point, target=target)

      update_runs_table(exp_name)

      # Check if we are done
      if are_we_done(func, target, exp_name):
        return
  except Exception as e:
    if verbose: print("Oops!", e.__class__, "occurred.")
    if verbose: print(e)
    if verbose: logging.exception("Something awful happened!")
  if verbose: print(optimizer.max)
  val = optimizer.max['target']
  save_results(val, exp_name)

# input types: {"fp", "exp"}
def optimize(shared_lib: str, input_type: str, num_inputs: int, splitting: str):
  global CUDA_LIB
  CUDA_LIB = shared_lib

  assert num_inputs >= 1 and num_inputs <= 3

  funcs_fp_1 = [function_max_inf_fp_1, function_min_inf_fp_1, function_max_under_fp_1, function_min_under_fp_1]
  funcs_exp_1 = [function_max_inf_exp_1, function_min_inf_exp_1, function_max_under_exp_1, function_min_under_exp_1]
  funcs_fp_2 = [function_max_inf_fp_2, function_min_inf_fp_2, function_max_under_fp_2, function_min_under_fp_2]
  funcs_exp_2 = [function_max_inf_exp_2, function_min_inf_exp_2, function_max_under_exp_2, function_min_under_exp_2]
  funcs_fp_3 = [function_max_inf_fp_3, function_min_inf_fp_3, function_max_under_fp_3, function_min_under_fp_3]
  funcs_exp_3 = [function_max_inf_exp_3, function_min_inf_exp_3, function_max_under_exp_3, function_min_under_exp_3]
    
  if input_type == 'fp':
    if num_inputs == 1:
      if splitting == 'whole':
        initialize()
        for b in bounds_fp_whole_1():
          for f in funcs_fp_1:
            exp_name = [shared_lib, input_type, 'b_whole']
            run_optimizer(b, f, '|'.join(exp_name))
      if splitting == 'two':
        initialize()
        for b in bounds_fp_two_1():
          for f in funcs_fp_1:
            exp_name = [shared_lib, input_type, 'b_two']
            run_optimizer(b, f, '|'.join(exp_name))
      if splitting == 'many':
        initialize()
        for b in bounds_fp_many_1():
          for f in funcs_fp_1:
            exp_name = [shared_lib, input_type, 'b_many']
            run_optimizer(b, f, '|'.join(exp_name))
          
    elif num_inputs == 2:
      if splitting == 'whole':
        initialize()
        for b in bounds_fp_whole_2():
          for f in funcs_fp_2:
            exp_name = [shared_lib, input_type, 'b_whole']
            run_optimizer(b, f, '|'.join(exp_name))
      if splitting == 'two':
        initialize()
        for b in bounds_fp_two_2():
          for f in funcs_fp_2:
            exp_name = [shared_lib, input_type, 'b_two']
            run_optimizer(b, f, '|'.join(exp_name))
      if splitting == 'many':
        initialize()
        for b in bounds_fp_many_2():
          for f in funcs_fp_2:
            exp_name = [shared_lib, input_type, 'b_many']
            run_optimizer(b, f,  '|'.join(exp_name))

    elif num_inputs == 3:
      if splitting == 'whole':
        initialize()
        for b in bounds_fp_whole_3():
          for f in funcs_fp_3:
            exp_name = [shared_lib, input_type, 'b_whole']
            run_optimizer(b, f, '|'.join(exp_name))
      if splitting == 'two':
        initialize()
        for b in bounds_fp_two_3():
          for f in funcs_fp_3:
            exp_name = [shared_lib, input_type, 'b_two']
            run_optimizer(b, f, '|'.join(exp_name))
      if splitting == 'many':
        initialize()
        for b in bounds_fp_many_3():
          for f in funcs_fp_3:
            exp_name = [shared_lib, input_type, 'b_many']
            run_optimizer(b, f,  '|'.join(exp_name))

  elif input_type == 'exp':
    if num_inputs == 1:
      if splitting == 'whole':
        initialize()
        for b in bounds_exp_whole_1():
          for f in funcs_exp_1:
            exp_name = [shared_lib, input_type, 'b_whole']
            run_optimizer(b, f, '|'.join(exp_name))
      if splitting == 'two':
        initialize()
        for b in bounds_exp_two_1():
          for f in funcs_exp_1:
            exp_name = [shared_lib, input_type, 'b_two']
            run_optimizer(b, f, '|'.join(exp_name))
      if splitting == 'many':
        initialize()
        for b in bounds_exp_many_1():
          for f in funcs_exp_1:
            exp_name = [shared_lib, input_type, 'b_many']
            run_optimizer(b, f, '|'.join(exp_name))
        
    elif num_inputs == 2:
      if splitting == 'whole':
        initialize()
        for b in bounds_exp_whole_2():
          for f in funcs_exp_2:
            exp_name = [shared_lib, input_type, 'b_whole']
            run_optimizer(b, f, '|'.join(exp_name))
      if splitting == 'two':
        initialize()
        for b in bounds_exp_two_2():
          for f in funcs_exp_2:
            exp_name = [shared_lib, input_type, 'b_two']
            run_optimizer(b, f, '|'.join(exp_name))
      if splitting == 'many':
        initialize()
        for b in bounds_exp_many_2():
          for f in funcs_exp_2:
            exp_name = [shared_lib, input_type, 'b_many']
            run_optimizer(b, f, '|'.join(exp_name))

    elif num_inputs == 3:
      if splitting == 'whole':
        initialize()
        for b in bounds_exp_whole_3():
          for f in funcs_exp_3:
            exp_name = [shared_lib, input_type, 'b_whole']
            run_optimizer(b, f, '|'.join(exp_name))
      if splitting == 'two':
        initialize()
        for b in bounds_exp_two_3():
          for f in funcs_exp_3:
            exp_name = [shared_lib, input_type, 'b_two']
            run_optimizer(b, f, '|'.join(exp_name))
      if splitting == 'many':
        initialize()
        for b in bounds_exp_many_3():
          for f in funcs_exp_3:
            exp_name = [shared_lib, input_type, 'b_many']
            run_optimizer(b, f, '|'.join(exp_name))
  else:
    print('Invalid input type!')
    exit()

#-------------- Results --------------
#lassen60_26904/cuda_code_acos.cu.so|fp|b_many :    [0, 0, 0, 0, 32]
#lassen60_26904/cuda_code_hypot.cu.so|fp|b_many :     [0, 0, 5, 0, 0]
def print_results(shared_lib: str, number_sampling, range_splitting):
  key = shared_lib+'|'+number_sampling+'|b_'+range_splitting
  fun_name = os.path.basename(shared_lib)
  print('-------------- Results --------------')
  print(fun_name)
  if key in results.keys():
    print('\tINF+:', results[key][0])
    print('\tINF-:', results[key][1])
    print('\tSUB-:', results[key][2])
    print('\tSUB-:', results[key][3])
    print('\tNaN :', results[key][4])
  else:
    print('\tINF+:', 0)
    print('\tINF-:', 0)
    print('\tSUB-:', 0)
    print('\tSUB-:', 0)
    print('\tNaN :', 0) 

  print('\tRuns:', runs_results[key])
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

# -------------------------------------------------------

if __name__ == '__main__':
  optimize()

