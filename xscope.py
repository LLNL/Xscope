#!/usr/bin/env python3

import argparse
import math
import subprocess
import socket
import os
import bo_analysis
import sys
import shutil

#------------------------------------------------------------------------------
# Globals
#------------------------------------------------------------------------------
compute_cap = 'sm_35'

#------------------------------------------------------------------------------
# Code generation functions
#------------------------------------------------------------------------------

# Generates CUDA code for a given math function
def generate_CUDA_code(fun_name: str, params: list, directory: str) -> str:
  file_name = 'cuda_code_'+fun_name+'.cu'
  with open(directory+'/'+file_name, 'w') as fd:
    fd.write('// Atomatically generated - do not modify\n\n')
    fd.write('#include <stdio.h>\n\n')
    fd.write('__global__ void kernel_1(\n')
    signature = ""
    param_names = ""
    for i in range(len(params)):
      if params[i] == 'double':
        signature += 'double x'+str(i)+','
        param_names += 'x'+str(i)+','
    fd.write('  '+signature)
    fd.write('double *ret) {\n')
    fd.write('   *ret = '+fun_name+'('+param_names[:-1]+');\n')
    fd.write('}\n\n')

    fd.write('extern "C" {\n')
    fd.write('double kernel_wrapper_1('+signature[:-1]+') {\n')
    fd.write('  double *dev_p;\n')
    fd.write('  cudaMalloc(&dev_p, sizeof(double));\n')
    fd.write('  kernel_1<<<1,1>>>('+param_names+'dev_p);\n')
    fd.write('  double res;\n')
    fd.write('  cudaMemcpy (&res, dev_p, sizeof(double), cudaMemcpyDeviceToHost);\n')
    fd.write('  return res;\n')
    fd.write('  }\n')
    fd.write(' }\n\n\n')
  return file_name

# Generates C++ code for a given math function
def generate_CPP_code(fun_name: str, params: list, directory: str) -> str:
  file_name = 'cpp_code_'+fun_name+'.cpp'
  with open(directory+'/'+file_name, 'w') as fd:
    fd.write('// Atomatically generated - do not modify\n\n')
    fd.write('#include <cmath>\n\n')
    fd.write('double cpp_kernel_1( ')
    signature = ""
    param_names = ""
    for i in range(len(params)):
      if params[i] == 'double':
        signature += 'double x'+str(i)+','
        param_names += 'x'+str(i)+','
    fd.write(signature[:-1]+') {\n')
    fd.write('   return '+fun_name+'('+param_names[:-1]+');\n')
    fd.write('}\n\n')
  return file_name

#------------------------------------------------------------------------------
# Compilation & running external programs
#------------------------------------------------------------------------------

def run_command(cmd: str):
  try:
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
  except subprocess.CalledProcessError as e:
    print(e.output)
    exit()

def compile_CUDA_code(file_name: str, d: str):
  global compute_cap 
  shared_lib = d+'/'+file_name+'.so'
  cmd = 'nvcc '+' -arch='+compute_cap+' '+d+'/'+file_name+' -o '+shared_lib+' -shared -Xcompiler -fPIC'
  print('Running:', cmd)
  run_command(cmd)
  return shared_lib

def compile_CPP_code(file_name: str, d: str):
  cmd = 'g++ '+d+'/'+file_name+' -o '+d+'/'+file_name+'.so -shared -fPIC'
  print('Running:', cmd)
  run_command(cmd)

#------------------------------------------------------------------------------
# File and directory creation 
#------------------------------------------------------------------------------

def dir_name():
  return '_tmp_'+socket.gethostname()+"_"+str(os.getpid())

def create_experiments_dir() -> str:
    p = dir_name()
    print("Creating dir:", p)
    try:
        os.mkdir(p)
    except OSError:
        print ("Creation of the directory %s failed" % p)
        exit()
    return p

#------------------------------------------------------------------------------
# Function Classes
#------------------------------------------------------------------------------
class SharedLib:
  def __init__(self, path, inputs):
    self.path = path
    self.inputs = int(inputs)

class FunctionSignature:
  def __init__(self, fun_name, input_types):
    self.fun_name = fun_name
    self. input_types = input_types

#------------------------------------------------------------------------------
# Main driver
#------------------------------------------------------------------------------

#FUNCTION:acos (double)
##FUNCTION:acosh (double)
#SHARED_LIB:./app_kernels/CFD_Rodinia/cuda_code_cfd.cu.so, N
#SHARED_LIB:./app_kernels/backprop_Rodinia/cuda_code_backprop.cu.so, N
def parse_functions_to_test(fileName):
  #function_signatures = []
  #shared_libs = []
  ret = []
  with open(fileName, 'r') as fd:
    for line in fd:
      # Comments
      if line.lstrip().startswith('#'):
        continue
      # Empty line
      if ''.join(line.split()) == '':
        continue

      if line.lstrip().startswith('FUNCTION:'):
        no_spaces = ''.join(line.split())
        signature = no_spaces.split('FUNCTION:')[1]
        fun = signature.split('(')[0]
        params = signature.split('(')[1].split(')')[0].split(',')
        ret.append(FunctionSignature(fun, params))
        #function_signatures.append((fun, params))

      if line.lstrip().startswith('SHARED_LIB:'):
        lib_path = line.split('SHARED_LIB:')[1].split(',')[0].strip()
        inputs = line.split('SHARED_LIB:')[1].split(',')[1].strip()
        #shared_libs.append((lib_path, inputs))
        ret.append(SharedLib(lib_path, inputs))

  #return (function_signatures, shared_libs)
  return ret

# Namespace(af='ei', function=['./function_signatures.txt'], number_sampling='fp', range_splitting='many', samples=30)
def areguments_are_valid(args):
  if args.af != 'ei' and args.af != 'ucb' and args.af != 'pi':
    return False
  if args.samples < 1:
    return False
  if args.range_splitting != 'whole' and args.range_splitting != 'two' and args.range_splitting != 'many':
    return False
  if args.number_sampling != 'fp' and args.number_sampling != 'exp':
    return False
  return True

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Xscope tool')
  parser.add_argument('function', metavar='FUNCTION_TO_TEST', nargs=1, help='Function to test (file or shared library .so)')
  parser.add_argument('-a', '--af', default='ei', help='Acquisition function: ei, ucb, pi')
  parser.add_argument('-n', '--number-sampling', default='fp', help='Number sampling method: fp, exp')
  parser.add_argument('-r', '--range-splitting', default='many', help='Range splitting method: whole, two, many')
  parser.add_argument('-s', '--samples', type=int, default=30, help='Number of BO samples (default: 30)')
  parser.add_argument('--random_sampling', action='store_true', help='Use random sampling')
  parser.add_argument('--random_sampling_unb', action='store_true', help='Use random sampling unbounded')
  parser.add_argument('-c', '--clean', action='store_true', help='Remove temporal directories (begin with _tmp_)')
  args = parser.parse_args()

  # --------- Cleaning -------------
  if (args.clean):
    print('Removing temporal dirs...')
    this_dir = './'
    for fname in os.listdir(this_dir):
      if fname.startswith("_tmp_"):
        #os.remove(os.path.join(my_dir, fname))
        shutil.rmtree(os.path.join(this_dir, fname))
    exit()

  # --------- Checking arguments for BO approach ---------
  if (not areguments_are_valid(args)):
    print('Invalid input!')
    parser.print_help()

  input_file = args.function[0]
  functions_to_test = []
  if input_file.endswith('.txt'):
    functions_to_test = parse_functions_to_test(input_file)
  else:
    exit()

  # Create directory to save experiments
  d = create_experiments_dir()

  # --------------- BO approach -----------------
  # Set BO  max iterations
  bo_analysis.set_max_iterations(args.samples)

  # Generate CUDA and compile them
  for i in functions_to_test:
    if type(i) is FunctionSignature:
      f = generate_CUDA_code(i.fun_name, i.input_types, d)
      shared_lib = compile_CUDA_code(f, d)
      num_inputs = len(i.input_types)
    elif type(i) is SharedLib:
      shared_lib = i.path
      num_inputs = i.inputs

    # Random Sampling
    if args.random_sampling or args.random_sampling_unb:
      print('******* RANDOM SAMPLING on:', shared_lib)
      # Total samples per each input depends on:
      # 18 ranges, 30 max samples (per range), n inputs
      inputs = num_inputs
      max_iters = 30 * int(math.pow(18, inputs))
      unbounded = False
      if args.random_sampling_unb:
        unbounded = True
      bo_analysis.optimize_randomly(shared_lib, inputs, max_iters, unbounded)
      bo_analysis.print_results_random(shared_lib)

    # Run BO optimization
    print('*** Running BO on:', shared_lib)
    bo_analysis.optimize(shared_lib,
                        args.number_sampling, 
                        num_inputs, 
                        args.range_splitting)
    bo_analysis.print_results(shared_lib, args.number_sampling, args.range_splitting)

