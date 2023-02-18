#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Numerical results of Frank-Wolfe algorithm for computing the continuous 
relaxation of M-DDF on the IEEE 118-instance 

For consistence, the Frank-Wolfe algorithm substract the output value by \logdet(C) 
"""

import pandas as pd
import numpy as np
import frank_wolfe

# assign local names
frankwolfe  = frank_wolfe.frankwolfe

#number of non-reference buses
n = 118-1 


# Frank-Wolfe Algorithm
loc = 0
df_fw = pd.DataFrame(columns=('n', 's', 'continuous M-DDF', 'time'))

for s in range(5, 20, 5): # set the values of s
    print("This is case ", loc+1)
    x,  mindual, time  = frankwolfe(n, s) 
    df_fw.loc[loc] = np.array([n, s,  mindual, time])
    loc = loc+1  

