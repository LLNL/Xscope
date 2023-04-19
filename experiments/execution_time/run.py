#!/usr/bin/env python3

import os
import ctypes
import time

CUDA_LIB = 'cuda_code.cu.so'

def call_GPU_kernel(x0, x1, i):
  global CUDA_LIB
  script_dir = os.path.abspath(os.path.dirname(__file__))
  lib_path = os.path.join(script_dir, CUDA_LIB)
  E = ctypes.cdll.LoadLibrary(lib_path)
  E.kernel_wrapper.restype = ctypes.c_double
  res = E.kernel_wrapper(ctypes.c_double(x0), ctypes.c_double(x1), ctypes.c_int(i))
  return res


def runCalls():
  x = 0.1

  # Warm-up calls (unused)
  call_GPU_kernel(x, x, 0)
  call_GPU_kernel(x, x, 1)
  call_GPU_kernel(x, x, 2)

  for i in range(0,59):
    t = time.process_time()
    r = call_GPU_kernel(x, x, i)
    elapsed_time = time.process_time() - t
    print('i', i, 'r', r, 'time', elapsed_time)

if __name__ == '__main__':
  print('Running calls...')
  runCalls()
