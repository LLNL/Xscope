#!/usr/bin/env python3.11

import math
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

# Time it takes to update BO
AVERAGE_BO_UPDATE_TIME = 0.027242362

exec_time_whole = {}
exec_time_two = {}
exec_time_many = {}

# exec_time_file
# acos  time  0.000318512
# acosh time  0.00029263
# asin  time  0.000294156
# asinh time  0.00030509

# runs_file
# acos fp whole 3
# acos fp two 7
# acos fp many 1031
# acos exp whole 7
# acos exp two 103
# acos exp many 415

def parse_data(exec_time_file, runs_file):
    exec_time = {}
    with open(exec_time_file, 'r') as fd:
        for l in fd:
            fun = l.split()[0]
            t = float(l.split()[2])
            exec_time[fun] = t
    #print(exec_time)
    
    with open(runs_file, 'r') as fd:
        for l in fd:
            if l.split()[1] == 'fp':
                fun = l.split()[0]
                type = l.split()[2]
                runs = int(l.split()[3])
                if type == 'whole':
                    exec_time_whole[fun] = (exec_time[fun]+AVERAGE_BO_UPDATE_TIME) * runs
                if type == 'two':
                    exec_time_two[fun] = (exec_time[fun]+AVERAGE_BO_UPDATE_TIME) * runs
                if type == 'many':
                    exec_time_many[fun] = (exec_time[fun]+AVERAGE_BO_UPDATE_TIME) * runs
                    
def plot_exec_time():
    fig, ax = plt.subplots()
    
    # --- Many -- 
    new_dict = dict(sorted(exec_time_many.items(), key=lambda item: item[1]))
    labels = list(new_dict.keys())    
    x = np.arange(len(list(new_dict.keys())))
    ax.plot(range(len(labels)), list(new_dict.values()), '.', label='Many')
    
    # --- Two --
    exec_time_two_values = []
    for k in labels:
        exec_time_two_values.append(exec_time_two[k])
    ax.plot(range(len(labels)), exec_time_two_values, '*', label='Two')

    # --- Whole -- 
    exec_time_whole_values = []
    for k in labels:
        exec_time_whole_values.append(exec_time_whole[k])
    ax.plot(range(len(labels)), exec_time_whole_values, '+', label='Whole')

    ax.legend()
    ax.set_ylabel('Time (sec)')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation='vertical', fontsize=10)
    fig = plt.gcf()
    fig.set_size_inches(12, 3)
    #plt.show()
    fig.savefig('exec_time.pdf', bbox_inches = "tight")
    

if __name__ == '__main__':
    exec_time_file = sys.argv[1]
    runs_file = sys.argv[2]
    print(exec_time_file, runs_file)
    parse_data(exec_time_file, runs_file)
    
    #print(len(exec_time_whole), len(exec_time_two), len(exec_time_many))
    plot_exec_time()
