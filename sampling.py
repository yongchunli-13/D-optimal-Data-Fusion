#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The implementation of the randomized sampling algorithm
"""

import numpy as np
import frank_wolfe
import user
import random
import datetime


# assign local names to functions in the user and frank_wolfe file
frankwolfe  = frank_wolfe.frankwolfe
f = user.f

def randrounding(xsol, n, s):
    np.random.seed(2)

    xsel=[]
    A=0.0
    B=0.0    
    A=calconvolve(xsel,xsol,n,s)
    for c in range(n):
        if sum(xsel)>=s:
            for i in range(n-c):
                xsel.append(0.0)
            break
        
        if c-sum(xsel)>=n-s:
            for i in range(n-c):
                xsel.append(1.0)
            break
        
        num=random.uniform(0, 1)
        
        xsel.append(1)
        B=calconvolve(xsel,xsol,n,s)
        if((xsol[c]*B)/A >=num):
            A=B
        else:
            A=A-xsol[c]*B
            xsel[c]=0.0
            
    return xsel, f(xsel)

################################
def calconvolve(xsel,xsol,n,s):
    
    l=len(xsel)
    nz=sum(k>0 for k in xsel)
    value=0.0
    acon=[1,xsol[l]]
    for i in range(n-l-1):
        acon=np.convolve(acon,[1,xsol[1+i+l]])

    value=acon[s-nz]

    return value


# Function randrounding needs input n, s, and N;
# outputs the solution, objective value and running time
def sampling(n, s, N):
    start = datetime.datetime.now()
    # run Frank-Wolfe to obtain an optimal solution of continuous M-DDF
    [xsol, primal, ftime]=frankwolfe(n, s)  
    print("The running time of Frank-Wolfe algorithm = ", ftime)
    print('The current objective value is:', primal)
    
    bestx = 0
    bestf = -1e+10
    for i in range(N):
        x,fval = randrounding(xsol,  n, s)
        if fval > bestf:         
            bestx = x
            bestf = fval
    
    end = datetime.datetime.now()
    time = (end-start).seconds
    print("The output of Sampling algorithm = ", bestf)
    return bestf, bestx, time
