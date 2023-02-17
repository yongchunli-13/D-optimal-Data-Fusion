#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Numerical results of Frank-Wolfe, local search and sampling Algorithms 
on the IEEE 118-instance 

For consistence, all algorithms substract the output value by \logdet(C) 
"""

import pandas as pd
import numpy as np
import local_search
import frank_wolfe
import sampling

# assign local names
localsearch  = local_search.localsearch
frankwolfe  = frank_wolfe.frankwolfe
sampling = sampling.sampling

#number of non-reference buses
n = 118-1 

# Local Search Algorithm
loc = 0
df_ls = pd.DataFrame(columns=('n', 's', 'objective value', 'time'))

for s in range(5, 20, 5): # set the values of s
    print("This is case ", loc+1)
    fval, xsol, time  = localsearch(n, s) 
    df_ls.loc[loc] = np.array([n, s, fval, time])
    loc = loc+1  


# Frank-Wolfe Algorithm
loc = 0
df_fw = pd.DataFrame(columns=('n', 's', 'continuous M-DDF', 'time'))

for s in range(5, 20, 5): # set the values of s
    print("This is case ", loc+1)
    x,  mindual, time  = frankwolfe(n, s) 
    df_fw.loc[loc] = np.array([n, s,  mindual, time])
    loc = loc+1  

# Sampling Algorithm
loc = 0
df_samp = pd.DataFrame(columns=('n', 's', 'objective value', 'time'))

N = 100 # the number of repetitions for sampling 
for s in range(5, 20, 5): # set the values of s
    print("This is case ", loc+1)
    fval, xsol, time  = sampling(n, s, N) 
    df_samp.loc[loc] = np.array([n, s, fval, time])
    loc = loc+1  
