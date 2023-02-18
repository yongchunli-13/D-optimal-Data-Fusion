#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Numerical results of optimality cuts and exact Branch-and-Bound algorithm
"""

import pandas as pd
import numpy as np
import frank_wolfe
import local_search
import branch_and_bound
import optcut
import datetime

# assign local names
frankwolfe  = frank_wolfe.frankwolfe
optcut = optcut.optcut
localsearch = local_search.localsearch
BranchBound = branch_and_bound.BranchBound


#number of non-reference buses
n = 118-1 

# Optimality Cuts
loc = 0
df_cut = pd.DataFrame(columns=('n', 's', '#a', '#b', '#c','#d', '#e', 'cut time'))

for s in range(5, 20, 5): # set the values of s
    print("This is case ", loc+1)
    cut_a, cut_b, cut_c, cut_d, cut_e, time = optcut(n, s)
    df_cut.loc[loc] = np.array([n, s, len(cut_a), len(cut_b), len(cut_c), len(cut_d), len(cut_e), time])
    loc = loc+ 1
    

# Exact B&B Algorithm
loc = 0
df = pd.DataFrame(columns=('n', 's', 'output_val', 'obj_val', 'ub', 'lb', 'time'))
for s in range(16, 20, 5): # set the values of s
    print("This is case ", loc+1)
    
    start = datetime.datetime.now()
    cut_a, cut_b, cut_c, cut_d, cut_e, time = optcut(n, s)

    fval, xsol, time = localsearch(cut_a, cut_b, n, s)
    
    temp = list(set(cut_a).union(set(cut_b)))
    rxsol =[]
    rset = list(set(range(n))-set(temp))
    for i in rset:
        rxsol.append(xsol[i])
    
    # output_val: the output value of B&B reached optimality or time limit
    # obj_val: the objective value of DDF at output solution of B&B reached optimality or time limite
    output_val, obj_val, ub, lb = BranchBound(rxsol, cut_a, cut_b, cut_c, cut_d, cut_e, n, s)
    
    end = datetime.datetime.now()
    time = (end-start).seconds
        
    df.loc[loc] = np.array([n, s, output_val, obj_val, ub, lb, time])
    loc = loc+1 

    