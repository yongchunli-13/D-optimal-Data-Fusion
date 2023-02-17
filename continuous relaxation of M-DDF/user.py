#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The user file includes the data preprocessing and function definitions 
including, objective function, rank-one update, and gradient calculation
"""


import os
import numpy as np
import pandas as pd
from math import log
from math import sqrt
from numpy import matrix
from numpy import array
import scipy

## Data preprocessing 
def gen_data(n): 
    global C
    global gamma
    
    
    ## input Fisher information matrix $C$ from existing sensors
    C = pd.read_csv(os.getcwd()+'/gain_matrix_118.csv',
                          encoding = 'utf-8',sep=',')
    C = array(C)
    C = np.delete(C, [0], axis=1) # delete index column   
    C = matrix(C)


    ## input PMU variance $\sigma$
    sigma = pd.read_csv(os.getcwd()+'/PMU_variance_118.csv',
                         encoding = 'utf-8',sep=',')
    sigma = array(sigma)
    sigma = np.delete(sigma, [0], axis=1) # delete index column 
    sigma = sigma*sigma
    gamma = 1/sigma  
    gamma = gamma[:,0]
    
    
## Cholesky decomposition to construct matrix $V$ used in M-DDF
def gen_data_mddf(n):
    global V
    
    E = np.eye(n, dtype=int) # identity matrix 
    
    tempC = matrix(scipy.linalg.sqrtm(C.I)) # compute the square root of C^-1
    
    B = [[sqrt(gamma[i])* (tempC[i]) for i in range(n)]]
    B = array(B[0])
    B = B.reshape(n,n)
    B = matrix(B)
    
    W = B*B.T
    
    V = np.linalg.cholesky(E + W) # Cholesky decomposition of I+BB^T
    V = matrix(V)
    
    intval = np.linalg.slogdet(C)[1] # logdet(C)
    
    return V, intval
   
 
## The objective function of DDF: \logdet(C+\sum_{i in S}gamma[i]*e_i*e_i^T)
def f(x):     
    sel = np.flatnonzero(x) 
    val = 0.0
    val = val + C
    for i in sel:
        val[i,i] = val[i,i] + gamma[i]
    fval = 0.0    
    fval = np.linalg.slogdet(val)[1] - np.linalg.slogdet(C)[1]
    return fval 

## initial parameters for greedy
def intgreedy(x):
    X = C
    Xs = C.I
    fval = np.linalg.slogdet(C)[1]
    return X, Xs, fval

# Update the inverse matrix by adding a rank-one matrix
def upd_inv_add(X, Xs, opti): 
    Y = 0
    Y = Y + X 
    Y[opti, opti] = Y[opti, opti] + gamma[opti]

    temp =  Xs[:,opti] * Xs[:,opti].T
    Ys = Xs - (gamma[opti]/(1+gamma[opti]*Xs[opti,opti]))* temp    
    return Y, Ys
 	
# Update the inverse matrix by minusing a rank-one matrix
def upd_inv_minus(X, Xs, opti):
    Y = 0 
    Y = Y + X 
    Y[opti, opti] = Y[opti, opti] - gamma[opti]       
    
    temp =  Xs[:,opti] * Xs[:,opti].T
    Ys = Xs + (gamma[opti]/(1-gamma[opti]*Xs[opti,opti]))* temp  
    return Y, Ys

## The rank-one update for greedy
def srankone(X, Xs, indexN, n, val):   
    opti = 0.0
    Y = 0.0
    
    temp = []
    for i in indexN:
        temp.append(1 + gamma[i]*Xs[i,i])
    tempi = np.argmax(temp)
    opti = indexN[tempi]
    maxf = temp[tempi]

    val  = val + log(maxf)
    
    Y,Ys = upd_inv_add(X,Xs,opti) # update X and Xs
    
    return Y,Ys,opti,val 


## The rank-one update for local search
def findopt(X, Xs, i, indexN, n,val):
    
    Y=0.0
    Ys=0.0

    Y, Ys = upd_inv_minus(X,Xs,i)
    
    nsel = len(indexN)
    temp = []
    for j in range(nsel):
        temp.append(gamma[indexN[j]]*Ys[indexN[j],indexN[j]])
    
    
    opti = indexN[np.argmax(temp)]
        
    val = val + log(max(1-gamma[i]*Xs[i,i], 1e-14)) + log(1+gamma[opti]*Ys[opti,opti])

    return Y, Ys, opti, val

#### Compute the supgradient of objective function of M-DDF ####
def find_k(x,s,d):
    for i in range(s-1):
        k = i
        mid_val = sum(x[j] for j in range(k+1,d))/(s-k-1)
        if mid_val >= x[k+1]-1e-14 and mid_val < x[k]+1e-14:    
            return k, mid_val
        

def grad_fw(fmat, x, s):
    nx = len(x)
    
    val = 0.0
    val = fmat
        
    [a,b] = np.linalg.eigh(val) 
    a = a.real # engivalues
    b = b.real # engivectors
    a[a<1e-14] = 0

    sorted_a = sorted(a, reverse=True)     
    k,nu = find_k(sorted_a,s,nx)
    
    fval = 0.0 
    for i in range(s):
        if i <= k:
            fval = fval + log(sorted_a[i])
        else:
            fval = fval + log(nu)
 
    tempval = 1/nu
    engi_val = [0]*nx
    
    for i in range(nx):
        if(a[i] > nu):
            engi_val[i] = 1/a[i]
        else:
            engi_val[i] = tempval

    W = b*np.diag(engi_val)*b.T 

    val = 0.0
    subg = [0.0]*nx
    
    temp = V*W*V.T
    val = temp.diagonal()
    val = val.reshape(nx,1)   
    val = list(val)
    
    for i in range(nx):
        subg[i] = val[i][0,0] # supgradient at x
        
    temp = np.array(subg)
    sindex = np.argsort(-temp)   
    y = [0]*nx
    for i in range(s):
        y[sindex[i]] = 1 # solution of linear oracle maximization

    # construct feasible dual solution
    nu = subg[sindex[s-1]]
    mu = [0]*nx
    for i in range(s):
        mu[sindex[i]] = subg[sindex[i]]- nu 
        
    dual_gap = s*nu+sum(mu)-s
        
    return fval, subg, y, dual_gap

      

    
 
