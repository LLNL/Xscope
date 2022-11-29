#!/usr/bin/env python3

import os
import stat
import math
import sys

# ================ Globals =================
# function tests will be divided by _nodes_
_nodes_ = 20 
_time_per_node_ = '01:00'
_af_ = 'ei'
_number_sampling_ = 'exp'
_splitting_ = 'many'
_samples_ = '25'
# ==========================================

def create_input_files(f: str):
  functions = []
  with open(f, 'r') as fd:
    for l in fd:
      if 'FUNCTION' in l:
        signature = l.split(':')[1][:-1]
        functions.append(signature)

  n = math.ceil(len(functions) / _nodes_)
  print('Funcs / Nodes:', n)
  for i in range(_nodes_):
    block = functions[i*n:i*n+n]
    print('Block', i+1, block)
    with open('_inputs_'+str(i+1)+'.txt', 'w') as fd:
      for b in block:
        fd.write('FUNCTION:'+b+'\n')


def create_job_scipts():
  for i in range(_nodes_):
    with open('script_'+str(i+1)+'.cmd', 'w') as fd:
      fd.write('#!/bin/bash\n')
      fd.write('#BSUB -nnodes 1\n')
      fd.write('#BSUB -q pbatch\n')
      fd.write('#BSUB -G asccasc\n')
      fd.write('#BSUB -N\n')
      fd.write('#BSUB -B\n')
      fd.write('#BSUB -W '+_time_per_node_+'\n')
      fd.write('#BSUB -e job_'+str(i+1)+'\n')
      fd.write('#BSUB -o job_'+str(i+1)+'\n')
      fd.write('\n')
      fd.write('date \n')
      fd.write('\n')
      fd.write('cd ..\n')
      fd.write('./xscope.py '+'-s '+_samples_+' -a '+_af_+' -n '+_number_sampling_+' -r '+_splitting_+' --save experiments/_inputs_'+str(i+1)+'.txt'+'\n')
      fd.write('\n')
      fd.write('date \n')
     
def create_submission_script():
  f = 'submit.sh'
  with open(f, 'w') as fd:
    fd.write('#!/bin/bash -x\n\n')
    for i in range(_nodes_):
      fd.write('bsub '+'script_'+str(i+1)+'.cmd\n')
  st = os.stat(f)
  os.chmod(f, st.st_mode | stat.S_IEXEC)

if __name__ == '__main__':
  input_file = sys.argv[1]
  print('Processing', input_file)

  create_input_files(input_file)
  create_job_scipts()
  create_submission_script()
