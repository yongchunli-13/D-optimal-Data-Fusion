#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The implementation of the greedy algorithm
"""

import user 
import numpy as np
import datetime

# assign local names to functions in the user file
gen_data = user.gen_data
srankone = user.srankone
intgreedy = user.intgreedy


## Function grd needs input n and s; outputs the objectve value, 
## solution, matrix C+\sum_{i in S}gamma[i]*e_i*e_i^T and its inverse, 
## and running time of the greedy algorithm

## Let subset $S_1$ denote selected points
## Let subset $S_0$ denote discarded points
def grd(S1, S0, n, s): 
    c = 1
    x = [0]*n # chosen set
    for i in S1:
        x[i] = 1
        
    y = [1]*n # unchosen set
    for i in S0:
        y[i] = 0
    for i in S1:
        y[i] = 0
    indexN = np.flatnonzero(y)
     
    gen_data(n) # load data
     
    index = 0
    X, Xs, fval = intgreedy(x)
    intval = fval
    Y = np.zeros([n,n])
    Ys = np.zeros([n,n])
    val = fval
    
    start = datetime.datetime.now()
 
    while c < s+1-len(S1):   
        Y, Ys, index, fval = srankone(X, Xs, indexN, n, val)   
        X = Y
        Xs = Ys
        val = fval         
        
        x[index] = 1
        y[index] = 0
        indexN = np.flatnonzero(y)    
        c = c + 1
        # print(index, fval)
        
    grdx = x # output solution of greedy
    grdf = val - intval # output value of greedy
    
    end = datetime.datetime.now()
    time = (end - start).seconds 
    
    return grdf, grdx, X, Xs, time 

