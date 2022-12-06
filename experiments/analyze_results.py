#!/usr/bin/env python3

import math
import pandas as pd
import sys
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from collections import defaultdict
from statistics import mean
from matplotlib import rcParams

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

runs_per_exp = {}
excpetions_per_exp = {}

# For multi-file analysis
total_exceptions_found = defaultdict(dict) # key is method
trials_to_trigger = defaultdict(dict)
functions_where_ex_found = defaultdict(dict)
number_distinct_exceptions = defaultdict(dict)

# random_exceptions_per_function[exp_name][function] = [0,1,0,0,0]
random_exceptions_per_function = defaultdict(dict)

# ------------------------------------------------------------------
# Function inputs
fun_inputs = {
'acos': 1,
'acosh': 1,
'asin': 1,
'asinh': 1,
'atan': 1,
'atan2': 2,
'atanh': 1,
'cbrt': 1,
'ceil': 1,
'copysign': 2,
'cos': 1,
'cosh': 1,
'cospi': 1,
'cyl_bessel_i1': 1,
'erf': 1,
'erfc': 1,
'erfcinv': 1,
'erfcx': 1,
'erfinv': 1,
'exp': 1,
'exp10': 1,
'exp2': 1,
'expm1': 1,
'fabs': 1,
'fdim': 2,
'floor': 1,
'fmax': 2,
'fmin': 2,
'fmod': 2,
'hypot': 2,
'j0': 1,
'j1': 1,
'lgamma': 1,
'log': 1,
'log10': 1,
'log1p': 1,
'log2': 1,
'logb': 1,
'max': 1,
'min': 1,
'nearbyint': 1,
'nextafter': 2,
'normcdf': 1,
'normcdfinv': 1,
'pow': 2,
'rcbrt': 1,
'remainder': 2,
'rhypot': 2,
'rint': 1,
'round': 1,
'rsqrt': 1,
'sin': 1,
'sinpi': 1,
'tan': 1,
'tanh': 1,
'tgamma': 1,
'trunc': 1,
'y0': 1,
'y1': 1,
}

def get_max_runs():
  global fun_inputs
  ret = defaultdict(int)
  bo_samples = 25
  ret['whole'] = bo_samples * len(fun_inputs)
  for k in fun_inputs:
    inputs  = fun_inputs[k]
    ret['two'] += bo_samples * int(math.pow(2,inputs))
    ret['fp_many'] += bo_samples * int(math.pow(16,inputs))
    ret['exp_many'] += bo_samples * int(math.pow(8,inputs))
  return ret
# ------------------------------------------------------------------


#lassen596_128115/cuda_code_rcbrt.cu.so|fp|b_whole 3
#lassen596_128115/cuda_code_rcbrt.cu.so|fp|b_two 7
#lassen596_128115/cuda_code_rcbrt.cu.so|fp|b_many 1423
#lassen726_116469/cuda_code_acos.cu.so|fp|b_whole :     [0, 0, 0, 0, 4]
#lassen596_128115/cuda_code_y1.cu.so|exp|b_many 707
def parse_runs(filename: str):
  with open(filename, 'r') as fd:
    for l in fd:
      if "cuda_code_" in l and not ":" in l:
        runs = l.split()[1]
        exp_name = l.split()[0]
        exp_name = exp_name.split('cuda_code_')[1]
        runs_per_exp[exp_name] = runs


# ====== Trace format ======
#_tmp_lassen257_171193/cuda_code_acos.cu.so|fp|b_many,0,0,0,0,8
#_tmp_lassen257_171193/cuda_code_acosh.cu.so|fp|b_many,0,0,0,0,14
#_tmp_lassen257_171193/cuda_code_asin.cu.so|fp|b_many,0,0,2,1,8
def parse_exceptions(filename: str):
  with open(filename, 'r') as fd:
    for l in fd:
      if "cuda_code_" in l or "cpu_code_" in l:
        exp_name = l.split(',')[0]
        exp_name = exp_name.split('_code_')[1]
        v = l.split(',')[1:]
        values = [eval(i) for i in v] # convert from string to int: '0' -> 0
        #values = json.loads(l.split(':')[1])
        excpetions_per_exp[exp_name] = values

# Convert lists to ones: [2,4,0] => [1,1,0]        
def convert_values_to_one(val_list):
  ret = []
  for i in range(len(val_list)):
    if val_list[i] != 0:
      ret.append(1)
    else:
      ret.append(0)
  return ret

# [1,0,0] + [0,1,0] => [1,1,0]
def mask_lists(list_a, list_b):
  new_list = []
  for i in range(len(list_a)):
    if list_a[i] != 0 or list_b[i] != 0:
      new_list.append(1)
    else:
      new_list.append(0)
  return new_list
  
def parse_exceptions_random_comparison(filename: str):
  global random_exceptions_per_function
  with open(filename, 'r') as fd:
    for l in fd:
      if "cuda_code_" in l and ":" in l:
        exp_name = l.split()[0]
        fun_name = exp_name.split('cuda_code_')[1].split('|')[0]
        exp_name = exp_name.split('cuda_code_')[1]
        exp_name = exp_name.split('.so|')[1]
        values = json.loads(l.split(':')[1])
        
        if exp_name == 'fp|b_many' or exp_name == 'RANDOM' or exp_name == 'RANDOM_stop':
          random_exceptions_per_function[exp_name][fun_name] = values
          
    for func in random_exceptions_per_function['RANDOM']:
      if func not in random_exceptions_per_function['fp|b_many']:
        random_exceptions_per_function['fp|b_many'][func] = [0,0,0,0,0]

def get_distinct_exepctions(values: list):
  ret = 0
  for i in values:
    if i != 0:
      ret += 1
  return ret

def plot_random_results():
  global random_exceptions_per_function
  print('In plot_random_results')
  x_random = []
  x_random_stop = []
  x_xscope = []
  labels = []
  for k in random_exceptions_per_function:
    for j in random_exceptions_per_function[k]:
      print(k, j, random_exceptions_per_function[k][j])
      n = get_distinct_exepctions(random_exceptions_per_function[k][j])
      if k == 'RANDOM':
        x_random.append(n)
        labels.append(j.split('.')[0])
      elif k == 'RANDOM_stop':
        x_random_stop.append(n)
      else:
        x_xscope.append(n)

  x = np.arange(len(x_random))  # the label locations
  width = 0.25  # the width of the bars
  fig, ax = plt.subplots()
  rects1 = ax.bar(x-width, x_random_stop, width, label='Random', color='cyan', hatch='.')
  rects2 = ax.bar(x, x_random, width, color='blue', label='Random_unbounded')
  rects3 = ax.bar(x+width, x_xscope, width, label='Xscope', color='orange', hatch='/')
  
  #rects3 = ax.bar(x+width, many_r_runs, width, label='Many')

  # Add some text for labels, title and custom x-axis tick labels, etc.
  ax.set_ylabel('Exception Types Found')
  #ax.set_yscale('log')
  ax.set_xticks(x)
  ax.set_xticklabels(labels, rotation='vertical', fontsize=10)
  ax.legend()
  #ax.bar_label(rects1, padding=3)
  #ax.bar_label(rects2, padding=3)
  
  fig.tight_layout()
  fig = plt.gcf()
  fig.set_size_inches(12, 3)
  ax.set_xlim([-1, len(x)])
  #plt.show()
  fig.savefig('random_comparison.pdf', bbox_inches = "tight")

# File names format:
#   results-[Acq. function]-[N samples]-[sampling method]-[splitting method].txt
# Examples:
#       results-EI-25-fp-many.txt
#       results-EI-30-fp-many.txt
# Trace example:
# _tmp_lassen344_58707/cuda_code_acos.cu.so|fp|b_many,0,0,0,0,8
# _tmp_lassen344_58707/cuda_code_acosh.cu.so|fp|b_many,0,0,0,0,14
def parse_exceptions_from_method(filename: str):
  global total_exceptions_found, trials_to_trigger, functions_where_ex_found, number_distinct_exceptions
  print('Parsing file:', filename)
  ac_function = filename.split('-')[1]
  samples = int(filename.split('-')[2])
  with open(filename, 'r') as fd:
    for l in fd:
      if "cuda_code_" in l:
        exp_name = l.split(',')[0]
        fun_name = exp_name.split('cuda_code_')[1].split('|')[0]
        exp_name = exp_name.split('cuda_code_')[1]
        exp_name = exp_name.split('.so|')[1]
        exp_name += '-'+ac_function
        # Get number of inputs found
        v = l.split(',')[1:]
        values = [eval(i) for i in v] # convert from string to int: '0' -> 0
        s = sum(values) # sum of all inputs
        
        print('exp_name', exp_name, 'fun_name', fun_name, 's', s, 'values', values)
        
        # total_exceptions_found values
        if exp_name not in total_exceptions_found[samples]:
          total_exceptions_found[samples][exp_name] = s
        else:
          total_exceptions_found[samples][exp_name] += s
    
        # functions_where_ex_found values
        if s > 0:
          if exp_name not in functions_where_ex_found[samples]:
            functions_where_ex_found[samples][exp_name] = set([fun_name])
          else:
            functions_where_ex_found[samples][exp_name].add(fun_name)
          
        # number_distinct_exceptions values
        if exp_name not in number_distinct_exceptions[samples]:
          normalized_values = convert_values_to_one(values)
          number_distinct_exceptions[samples][exp_name] = normalized_values
        else:
          normalized_values = convert_values_to_one(values)
          new_list = mask_lists(normalized_values, number_distinct_exceptions[samples][exp_name])
          number_distinct_exceptions[samples][exp_name] = new_list
          
      # trials_to_trigger values
      #if "cuda_code_" in l and "%" in l:
      #  exp_name = l.split()[0]
      #  exp_name = exp_name.split('cuda_code_')[1]
      #  exp_name = exp_name.split('.so|')[1]
      #  exp_name += '-'+ac_function
      #  val = int(l.split('%')[1])
        
      #  if exp_name not in trials_to_trigger[samples]:
      #    trials_to_trigger[samples][exp_name] = [val]
      #  else:
      #    trials_to_trigger[samples][exp_name].append(val)
          
# Normalize results by max samples:
# whole: 30
# two: 2*30
# fp_many: 30*18 = 540
# exp_many: 30*8 = 240
def plot_multi_files():
  global total_exceptions_found, trials_to_trigger, functions_where_ex_found, number_distinct_exceptions
  
  SMALL_SIZE = 10
  MEDIUM_SIZE = 12
  BIGGER_SIZE = 12
  #plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
  plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
  plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
  plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
  plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
  plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
  plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
  
  # -------- Plot Total Exceptions Found ------------------
  x = []
  fp_b_whole_EI = []
  fp_b_two_EI = []
  fp_b_many_EI = []
  exp_b_whole_EI = []
  exp_b_two_EI = []
  exp_b_many_EI = []
  for k in total_exceptions_found:
    x.append(k)
    for j in total_exceptions_found[k]:
      v = total_exceptions_found[k][j]
      if 'fp|b_whole-EI' == j:
        fp_b_whole_EI.append(v/get_max_runs()['whole'])
      if 'fp|b_two-EI' == j:
        fp_b_two_EI.append(v/get_max_runs()['two'])
      if 'fp|b_many-EI' == j:
        fp_b_many_EI.append(v/get_max_runs()['fp_many'])
      if 'exp|b_whole-EI' == j:
        exp_b_whole_EI.append(v/get_max_runs()['whole'])
      if 'exp|b_two-EI' == j:
        exp_b_two_EI.append(v/get_max_runs()['two'])
      if 'exp|b_many-EI' == j:
        exp_b_many_EI.append(v/get_max_runs()['exp_many'])
        
  fig, ax = plt.subplots(1,2)
  xs, ys = zip(*sorted(zip(x, fp_b_whole_EI))) # sorting
  fig1 = ax[0].plot(xs, ys, 'o--',label='fp_whole')
  xs, ys = zip(*sorted(zip(x, fp_b_two_EI)))
  fig2 = ax[0].plot(xs, ys, '-^', label='fp_two')
  xs, ys = zip(*sorted(zip(x, fp_b_many_EI)))
  fig3 = ax[0].plot(xs, ys, 's-.', label='fp_many')
  xs, ys = zip(*sorted(zip(x, exp_b_whole_EI)))
  fig4 = ax[0].plot(xs, ys, '^:', label='exp_whole')
  xs, ys = zip(*sorted(zip(x, exp_b_two_EI)))
  fig5 = ax[0].plot(xs, ys, 'o-', label='exp_two')
  xs, ys = zip(*sorted(zip(x, exp_b_many_EI)))
  fig6 = ax[0].plot(xs, ys, 's:', label='exp_many')
  ax[0].set_ylabel('Total Inputs Found (Normalized)')
  ax[0].set_xlabel('Max Iterations')
  ax[0].legend()
  ax[0].set_title('(a)')
  #fig.tight_layout()
  #plt.show()
  #fig.savefig('total_exceptions_found.pdf')
  
  """
  # -------- Plot Trials to Trigger ------------------
  x = []
  fp_b_whole_EI = []
  fp_b_two_EI = []
  fp_b_many_EI = []
  exp_b_whole_EI = []
  exp_b_two_EI = []
  exp_b_many_EI = []
  for k in trials_to_trigger:
    x.append(k)
    for j in trials_to_trigger[k]:
      #print(k, j, trials_to_trigger[k][j])
      v = mean(trials_to_trigger[k][j])
      if 'fp|b_whole-EI' == j:
        fp_b_whole_EI.append(v)
      if 'fp|b_two-EI' == j:
        fp_b_two_EI.append(v)
      if 'fp|b_many-EI' == j:
        fp_b_many_EI.append(v)
      if 'exp|b_whole-EI' == j:
        exp_b_whole_EI.append(v)
      if 'exp|b_two-EI' == j:
        exp_b_two_EI.append(v)
      if 'exp|b_many-EI' == j:
        exp_b_many_EI.append(v)
        
  #fig, ax = plt.subplots()
  fig1 = ax[1].plot(x, fp_b_whole_EI, label='fp_whole_EI')
  fig2 = ax[1].plot(x, fp_b_two_EI, label='fp_two_EI')
  fig3 = ax[1].plot(x, fp_b_many_EI, label='fp_many_EI')
  fig4 = ax[1].plot(x, exp_b_whole_EI, label='exp_whole_EI')
  fig5 = ax[1].plot(x, exp_b_two_EI, label='exp_two_EI')
  fig6 = ax[1].plot(x, exp_b_many_EI, label='exp_many_EI')
  #ax[1].set_ylabel('Trials to Trigger')
  #ax[1].set_xlabel('Max Iterations')
  ax[1].legend()
  fig.tight_layout()
  plt.show()
  """
  
  # -------- Plot Functions with exceptions ------------------
  x = []
  fp_b_whole_EI = []
  fp_b_two_EI = []
  fp_b_many_EI = []
  exp_b_whole_EI = []
  exp_b_two_EI = []
  exp_b_many_EI = []
  for k in functions_where_ex_found:
    x.append(k)
    for j in functions_where_ex_found[k]:
      #print(k, j, functions_where_ex_found[k][j])
      v = len(functions_where_ex_found[k][j])
      if 'fp|b_whole-EI' == j:
        fp_b_whole_EI.append(v)
      if 'fp|b_two-EI' == j:
        fp_b_two_EI.append(v)
      if 'fp|b_many-EI' == j:
        fp_b_many_EI.append(v)
      if 'exp|b_whole-EI' == j:
        exp_b_whole_EI.append(v)
      if 'exp|b_two-EI' == j:
        exp_b_two_EI.append(v)
      if 'exp|b_many-EI' == j:
        exp_b_many_EI.append(v)
        
  #fig, ax = plt.subplots()
  xs, ys = zip(*sorted(zip(x, fp_b_whole_EI))) # sorting
  fig1 = ax[1].plot(xs, ys, 'o--', label='fp_whole')
  xs, ys = zip(*sorted(zip(x, fp_b_two_EI))) # sorting
  fig2 = ax[1].plot(xs, ys, '-^', label='fp_two')
  xs, ys = zip(*sorted(zip(x, fp_b_many_EI))) # sorting
  fig3 = ax[1].plot(xs, ys, 's-.', label='fp_many')
  xs, ys = zip(*sorted(zip(x, exp_b_whole_EI))) # sorting
  fig4 = ax[1].plot(xs, ys, '^:', label='exp_whole')
  xs, ys = zip(*sorted(zip(x, exp_b_two_EI))) # sorting
  fig5 = ax[1].plot(xs, ys, 'o-', label='exp_two')
  xs, ys = zip(*sorted(zip(x, exp_b_many_EI))) # sorting
  fig6 = ax[1].plot(xs, ys, 's:', label='exp_many')
  ax[1].set_ylabel('Exception-Triggering Functions')
  ax[1].set_xlabel('Max Iterations')
  ax[1].set_title('(b)')
  ax[1].legend(loc='center')
  fig.tight_layout()
  #plt.show()
  fig.savefig('max_iterations_evaluation.pdf')
  
  """
  # -------- Plot Distinct Exceptions Found ------------------
  x = []
  fp_b_whole_EI = []
  fp_b_two_EI = []
  fp_b_many_EI = []
  exp_b_whole_EI = []
  exp_b_two_EI = []
  exp_b_many_EI = []
  for k in number_distinct_exceptions:
    x.append(k)
    for j in number_distinct_exceptions[k]:
      #print(k, j, number_distinct_exceptions[k][j])
      v = sum(number_distinct_exceptions[k][j])
      if 'fp|b_whole-EI' == j:
        fp_b_whole_EI.append(v)
      if 'fp|b_two-EI' == j:
        fp_b_two_EI.append(v)
      if 'fp|b_many-EI' == j:
        fp_b_many_EI.append(v)
      if 'exp|b_whole-EI' == j:
        exp_b_whole_EI.append(v)
      if 'exp|b_two-EI' == j:
        exp_b_two_EI.append(v)
      if 'exp|b_many-EI' == j:
        exp_b_many_EI.append(v)
        
  fig, ax = plt.subplots()
  fig1 = ax.plot(x, fp_b_whole_EI, label='fp_whole_EI')
  fig2 = ax.plot(x, fp_b_two_EI, label='fp_two_EI')
  fig3 = ax.plot(x, fp_b_many_EI, label='fp_many_EI')
  fig4 = ax.plot(x, exp_b_whole_EI, label='exp_whole_EI')
  fig5 = ax.plot(x, exp_b_two_EI, label='exp_two_EI')
  fig6 = ax.plot(x, exp_b_many_EI, label='exp_many_EI')
  ax.set_ylabel('Distinct Exceptions Found')
  ax.set_xlabel('Max Iterations')
  ax.legend()
  fig.tight_layout()
  plt.show()
  """

# Changes values to 1
def convert_to_string(l: list):
  v = []
  for i in l:
    v.append(str(i))
    #if i != 0:
    #  v.append('1')
    #else:
    #  v.append('0')
  ret = ',' + ','.join(v)
  return ret
  
def plot_runs():
  whole_r_runs = []
  two_r_runs = []
  many_r_runs = []
  labels = []
  for exp_name in runs_per_exp.keys():
    r = runs_per_exp[exp_name]
    if '|exp|' in exp_name:
    #if '|fp|' in exp_name:
      if '|b_whole' in exp_name:
        whole_r_runs.append(int(r))
        labels.append(get_func_name(exp_name))
      elif '|b_two' in exp_name:
        two_r_runs.append(int(r))
      elif '|b_many' in exp_name:
        many_r_runs.append(int(r))
        
  #labels = ['G1', 'G2', 'G3', 'G4', 'G5']
  #men_means = [20, 34, 30, 35, 27]
  #women_means = [25, 32, 34, 20, 25]

  x = np.arange(len(whole_r_runs))  # the label locations
  width = 0.25  # the width of the bars
  fig, ax = plt.subplots()
  rects1 = ax.bar(x-width, whole_r_runs, width, label='Whole')
  rects2 = ax.bar(x, two_r_runs, width, label='Two')
  rects3 = ax.bar(x+width, many_r_runs, width, label='Many')

  # Add some text for labels, title and custom x-axis tick labels, etc.
  ax.set_ylabel('Runs')
  ax.set_yscale('log')
  ax.set_xticks(x)
  ax.set_xticklabels(labels, rotation='vertical', fontsize=10)
  ax.legend()
  #ax.bar_label(rects1, padding=3)
  #ax.bar_label(rects2, padding=3)
  
  fig.tight_layout()
  fig = plt.gcf()
  fig.set_size_inches(12, 3)
  ax.set_xlim([-1, len(x)])
  #plt.show()
  fig.savefig('runs_exp_method.pdf', bbox_inches = "tight")
  #fig.savefig('runs_fp_method.pdf', bbox_inches = "tight")


# acos.cu.so|fp|b\_many
def get_func_name(exp_name: str):
  return exp_name.split('|')[0].split('.')[0].replace('_','\_')

#def print_latex_table(file_csv: str, b_type: str, v_type: str):
#  df = pd.read_csv(file_csv)
#  df['sums'] = df.sum(axis=1)
#  print('\\begin{table*}')
#  print('\\caption{'+b_type.replace('_','')+'-'+v_type+'}')
#  print('\\begin{tabular}{|c|c|c|c|c|c|c|}')
#  print('Exp & INF+ & INF- & Sub+ & Sub- & NaN & Tot\\\\')
#  for row in df.itertuples():
#    print(row[1], '&', row[2], '&', row[3], '&', row[4], '&', row[5], '&', row[6], '&', row[7], '\\\\')

#  df_t = df.sum(axis=0)
#  print('Total &', df_t['INF_p'], '&', df_t['INF_n'], '&', df_t['Sub_p'], '&', df_t['Sub_n'], '&', df_t['NaN'], '\\\\')
#  print('\\end{tabular}')
#  print('\\end{table*}\n\n')

def print_latex_table(file_csv: str, b_type: str, v_type: str):
  df = pd.read_csv(file_csv)
  #df['sums'] = df.sum(axis=1)
  print('\\caption{'+b_type.replace('_','')+'-'+v_type+'}')
  print('\\scalebox{0.8}{')
  print('\\begin{tabular}{|c|c|c|c|c|c|}')
  print('Fun & INF+ & INF- & S+ & S- & NaN\\\\')
  for row in df.itertuples():
    print(row[1], '&', row[2], '&', row[3], '&', row[4], '&', row[5], '&', row[6], '\\\\')

  df_t = df.sum(axis=0)
  print('Total &', df_t['INF_p'], '&', df_t['INF_n'], '&', df_t['Sub_p'], '&', df_t['Sub_n'], '&', df_t['NaN'], '\\\\')
  print('\\end{tabular}')
  print('}')
  print('\\quad')

def print_short_latex_table(file_csv: str, b_type: str, v_type: str):
  df = pd.read_csv(file_csv)
  print('\\caption{'+b_type.replace('_','')+'-'+v_type+'}')
  print('\\scalebox{0.8}{')
  print('\\begin{tabular}{|c|c|c|c|c|}')
  print('INF+ & INF- & S+ & S- & NaN\\\\')
  for row in df.itertuples():
    print(row[2], '&', row[3], '&', row[4], '&', row[5], '&', row[6], '\\\\')

  df_t = df.sum(axis=0)
  print(df_t['INF_p'], '&', df_t['INF_n'], '&', df_t['Sub_p'], '&', df_t['Sub_n'], '&', df_t['NaN'], '\\\\')
  print('\\end{tabular}')
  print('}')
  print('\\quad')

def save_results_in_csv(file_csv: str, b_type: str, v_type: str):
  fd = open(file_csv, 'w')
  fd.write('Func,INF_p,INF_n,Sub_p,Sub_n,NaN\n')
  for exp_name in runs_per_exp.keys():
    runs = runs_per_exp[exp_name]   
    if v_type in exp_name:
      if b_type in exp_name:
        if exp_name not in excpetions_per_exp.keys():
          fd.write(get_func_name(exp_name) + ',0,0,0,0,0\n')
          continue
        values = excpetions_per_exp[exp_name]
        fd.write(get_func_name(exp_name) + convert_to_string(values) + '\n')
  fd.close()

  #if b_type == '|b_whole':
  #  print_latex_table(file_csv, b_type, v_type)
  #else:
  print_short_latex_table(file_csv, b_type, v_type)
  
def print_results():

  """
  # ---------- FP inputs ---------------
  print('\\begin{table*}')
  print('\\renewcommand{\\arraystretch}{0.8}')
  # *** whole ***
  file_csv = './fp_whole_results.csv'
  save_results_in_csv(file_csv, '|b_whole', '|fp|')

  # ** Two **
  file_csv = './fp_two_results.csv'
  save_results_in_csv(file_csv, '|b_two', '|fp|')
  
  # ** Many **
  file_csv = './fp_many_results.csv'
  save_results_in_csv(file_csv, '|b_many', '|fp|')
  print('\\end{table*}\n\n')
  """

  # ---------- Exp inputs ---------------
  print('\\begin{table*}')
  print('\\renewcommand{\\arraystretch}{0.8}')
  # *** whole ***
  file_csv = './exp_whole_results.csv'
  save_results_in_csv(file_csv, '|b_whole', '|exp|')

  # ** Two **
  file_csv = './exp_two_results.csv'
  save_results_in_csv(file_csv, '|b_two', '|exp|')
  
  # ** Many **
  file_csv = './exp_many_results.csv'
  save_results_in_csv(file_csv, '|b_many', '|exp|')
  print('\\end{table*}\n\n')
  
def print_AC_function_comparison():
  global total_exceptions_found, functions_where_ex_found, number_distinct_exceptions
  samples = '30'
  #print(total_exceptions_found)
  for k in total_exceptions_found[samples]:
    method = k.split('-')[0]
    AF = k.split('-')[1] # acq func
    if method == 'fp|b_many':
      print('total_exceptions_found', total_exceptions_found[samples][k])
      print('functions_where_ex_found', len(functions_where_ex_found[samples][k]))
      print('number_distinct_exceptions', sum(number_distinct_exceptions[samples][k]))
  
def parse_single_result_file(results):
    parse_runs(results)
    parse_exceptions(results)
    plot_runs()
    
# This parser results from different methods and sample sizes
def parse_multi_results_file(argv):
  for i in range(1,len(argv)):
    filename = argv[i]
    print(filename)
    parse_exceptions_from_method(filename)
    #parse_exceptions_random_comparison(filename)
    
def plot_main_experiments_results(argv):
  global excpetions_per_exp
  
  fig, ax = plt.subplots(4,1)
  fig.tight_layout()
  fig = plt.gcf()
  fig.set_size_inches(14, 12)

  for i in range(1,len(argv)):
    filename = argv[i]
    print('Parsing...',filename)
    parse_exceptions(filename)
    excpetions_per_exp = dict(sorted(excpetions_per_exp.items()))
    
    x = [] # functions
    inf_neg_list = []
    inf_pos_list = []
    sub_neg_list = []
    sub_pos_list = []
    nan_list = []
    for experiment in excpetions_per_exp:
      values = excpetions_per_exp[experiment]
      inf_neg_list.append(values[0])
      inf_pos_list.append(values[1])
      sub_neg_list.append(values[2])
      sub_pos_list.append(values[3])
      nan_list.append(values[4])
      fun = experiment.split('|')[0].split('.')[0]
      x.append(fun)
      num_sampling = experiment.split('|')[1]
      input_splitting = experiment.split('|')[2]
      
    # --- plot ----
    width = 0.35
    #fig, ax = plt.subplots()
    fig1 = ax[i-1].bar(x, inf_neg_list, width, label='INF-', hatch="||")
    fig2 = ax[i-1].bar(x, inf_pos_list, width, label='INF+', hatch="+")
    fig3 = ax[i-1].bar(x, sub_neg_list, width, label='SUB-', hatch="*")
    fig4 = ax[i-1].bar(x, sub_pos_list, width, label='SUB+', hatch="o")
    fig5 = ax[i-1].bar(x, nan_list, width, label='NaN', hatch="xx")

    sampling_method = experiment.split('|')[1]
    input_splitting = experiment.split('|')[2].split('_')[1]
    ax[i-1].set_title(sampling_method+' method / '+input_splitting+'-range')
    ax[i-1].legend()
    ax[i-1].set_ylabel('Inputs Found')
    ax[i-1].set_yscale('log')
    #ax[i-1].set_xticks(x)
    ax[i-1].set_xticklabels(x, rotation='vertical', fontsize=10)
    ax[i-1].set_ylim([-1, 200])
      
    # Clean up
    excpetions_per_exp = {}
  #plt.show()
  fig.savefig('gpu_results.pdf', bbox_inches = "tight")
  
def plot_CPU_experiments_results(filename):
  global excpetions_per_exp
  fig, ax = plt.subplots(1,1)
  fig.tight_layout()
  fig = plt.gcf()
  fig.set_size_inches(12, 4)

  #for i in range(1,len(argv)):
  #filename = argv[i]
  print('Parsing...',filename)
  parse_exceptions(filename)  
  excpetions_per_exp = dict(sorted(excpetions_per_exp.items()))
    
  x = [] # functions
  inf_neg_list = []
  inf_pos_list = []
  sub_neg_list = []
  sub_pos_list = []
  nan_list = []
  for experiment in excpetions_per_exp:
    values = excpetions_per_exp[experiment]
    inf_neg_list.append(values[0])
    inf_pos_list.append(values[1])
    sub_neg_list.append(values[2])
    sub_pos_list.append(values[3])
    nan_list.append(values[4])
    fun = experiment.split('|')[0].split('.')[0]
    x.append(fun)
    num_sampling = experiment.split('|')[1]
    input_splitting = experiment.split('|')[2]
    exp_name = experiment
    
  # --- plot ----
  width = 0.35
  #fig, ax = plt.subplots()
  fig1 = ax.bar(x, inf_neg_list, width, label='INF-', hatch="||")
  fig2 = ax.bar(x, inf_pos_list, width, label='INF+', hatch="+")
  fig3 = ax.bar(x, sub_neg_list, width, label='SUB-', hatch="*")
  fig4 = ax.bar(x, sub_pos_list, width, label='SUB+', hatch="o")
  fig5 = ax.bar(x, nan_list, width, label='NaN', hatch="xx")
  sampling_method = experiment.split('|')[1]
  input_splitting = experiment.split('|')[2].split('_')[1]
  ax.set_title(sampling_method+' method / '+input_splitting+'-range')
  ax.legend()
  ax.set_ylabel('Inputs Found')
  ax.set_yscale('log')
  #ax[i-1].set_xticks(x)
  ax.set_xticklabels(x, rotation='vertical', fontsize=10)
  ax.set_ylim([-1, 200])

  #plt.show()
  fig.savefig('cpu_results.pdf', bbox_inches = "tight")
  
if __name__ == '__main__':
  #results = sys.argv[1]
  #parse_single_result_file(results)
  
  # ---- Plot main results with all execptions -----
  # Input: multiple files
  #plot_main_experiments_results(sys.argv)
  
  # ----- Plot CPU experiments from Quarts (Intel) ---
  # Input: 1 file
  filename = sys.argv[1]
  plot_CPU_experiments_results(filename)

  #print_results()
  #print(get_max_runs())
  #exit()

  # ---- Plots Figure with normalized results (from all runs) ----
  #parse_multi_results_file(sys.argv)
  #plot_multi_files()
    
  #plot_random_results()
  #print_AC_function_comparison()
  
