#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The user file includes the data preprocessing and function definitions 
including, objective function, rank-one update, gradient calculation, etc.
"""


import os
import numpy as np
import scipy
import math
from math import log
from math import sqrt
from numpy import matrix
import pandas as pd


## Data preprocessing 
def gen_data(n):
    global S
    global C
    global gamma
    
    ## input Fisher information matrix C from existing sensors
    C = pd.read_csv(os.getcwd()+'/gain_matrix_118.csv',
                          encoding = 'utf-8',sep=',')
    C = np.array(C)
    C = np.delete(C, [0], axis=1) # delete index column   
    C = matrix(C)


    ## input PMU variance 
    sigma = pd.read_csv(os.getcwd()+'/PMU_variance_118.csv',
                         encoding = 'utf-8',sep=',')

    sigma = np.array(sigma)
    sigma = np.delete(sigma, [0], axis=1) # delete index column 
    sigma = sigma*sigma
    gamma = 1/sigma  
    gamma = gamma[:,0]
    
    temp = C
    a, b = np.linalg.eigh(temp.I)
    sqrta = [0]*n
    for i in range(n):
        sqrta[i] = sqrt(a[i])
    
    tempC = b*np.diag(sqrta)*b.T  # compute the square root of C^-1
    
    # construct matrix $B$ used in R-DDF
    B = [sqrt(gamma[i])* (tempC[i]) for i in range(n)] 
    B = np.array(B)
    B = B.reshape(n,n)
    B = matrix(B)
    
    S = [[(B[i].T * B[i]) for i in range(n)]] # change the set
    S = S[0]


## Cholesky decomposition to construct matrix $V$ used in M-DDF
def gen_data_mddf(n):
    global V
    global E
    
    E = np.eye(n, dtype=int) # identity matrix 
    
    temp = C
    a, b = np.linalg.eigh(temp.I)
    sqrta = [0]*n
    for i in range(n):
        sqrta[i] = sqrt(a[i])
    
    tempC = b*np.diag(sqrta)*b.T  # compute the square root of C^-1
    
    B = [[sqrt(gamma[i])* (tempC[i]) for i in range(n)]]
    B = np.array(B[0])
    B = B.reshape(n,n)
    B = matrix(B)
    
    W = B*B.T
    
    V = np.linalg.cholesky(E + W) # Cholesky decomposition of I+BB^T
    V = matrix(V)
    
    intval = np.linalg.slogdet(C)[1] # logdet(C)
    
    return V, intval    
   
## The objective function of DDF: \logdet(I +\sum_{i in S}b_i*b_i^T) 
def f(x):	
    nx = len(x)	
    val = 0.0
    sel = [i for i in range(nx) if x[i]>0.5]
    for i in sel:
        val = val + S[i]
    val = val + E
    val = np.linalg.slogdet(val) 
    return val[1]

## The objective function of Restricted DDF
## Let subset $S_1$ denotes selected points
def rf(x, S1, rset):	
    nx = len(x)	
    val = 0.0
    sel = [i for i in range(nx) if x[i]>0.5]
    for i in sel:
        val = val + S[rset[i]]
    for i in S1:
        val = val + S[i]
    val = val + E
    val = np.linalg.slogdet(val) 
    return val[1]


############## Define Functions for B&B Algorithm ############## 

## Function used for gradient inequalities given Resrticted DDF
## Let subset $S_1$ denote selected points
def grad_ineq(x, S1, rset, n):
    nx = len(x)

    val = 0.0
    sel = [i for i in range(nx) if x[i]>0.5]
    for i in sel:
        val = val + x[i]*S[rset[i]]	
    for i in S1:
        val = val + S[i]
    val = val + E
    
    # gradient of Resrticted R-DDF at x
    g = [0]*nx
    for i in range(nx):
        g[i] = np.matrix.trace(np.linalg.inv(val)*S[rset[i]])
        
    
    sel = [i for i in range(nx) if x[i]>0.5]
    mval = 0.0
    for i in sel:
        mval = mval + x[i]*V[rset[i]].T*V[rset[i]]
    for i in S1:
        mval = mval + V[i].T*V[i]
    
    [a,b] = np.linalg.eigh(mval) 
    a = a.real # engivalues
    b = b.real # engivectors
    a[a<1e-14] = 0

    sorted_a = sorted(a, reverse=True)   
    s = len(S1) + len(sel)
    nu = sorted_a[s-1]

    engi_val = [0]*n
    for i in range(n):
        if(a[i] > nu):
            engi_val[i] = 1/a[i]
        else: 
            engi_val[i] = 1/nu
            
    W = b*np.diag(engi_val)*b.T 

    # supgradient of Resrticted M-DDF at x
    subg = [0.0]*nx
    for i in range(nx):
        subg[i] =  np.matrix.trace(W*(V[rset[i]].T*V[rset[i]]))[0,0] 
            
    return g, subg


### Function used for gradient inequalities given Resrticted DDF
## Let subset $S_1$ denote selected points
def rhoNi(nx, S1, rset, n):
    CN = np.zeros((n, n))
    CN = CN + C
    for i in S1:
        CN[i,i] = CN[i,i] + gamma[i]
    for i in range(nx):
        CN[rset[i],rset[i]] = CN[rset[i],rset[i]] + gamma[rset[i]]
        
    CNinv = CN.I
    pho = [0]*nx
    for i in range(nx):
        pho[i] = -math.log(1-gamma[rset[i]]*CNinv[rset[i],rset[i]])
        
    return pho

def rhoE(nx, S1, rset, n):
    CN = np.zeros((n, n))
    CN = CN + C
    for i in S1:
        CN[i,i] = CN[i,i] + gamma[i]
    
    Cinv = CN.I
    
    pho = [0]*nx
    for i in range(nx):
        pho[i] = math.log(1+gamma[rset[i]]*Cinv[rset[i],rset[i]])
        
    return pho

def rhoS(x, S1, rset, n):
    sel = np.flatnonzero(x)
    nx = len(x)
    T = [i for i in range(nx) if x[i] < 0.5]

    CN = np.zeros((n, n))
    CN = CN + C
    for i in S1:
        CN[i,i] = CN[i,i] + gamma[i]
    for i in sel:
        CN[rset[i],rset[i]] = CN[rset[i],rset[i]] + gamma[rset[i]]
        
    CSinv = 0.0
    CSinv = CN.I
    
    phoSi = [0]*nx
    for i in sel:
        phoSi[i] = -math.log(1-gamma[rset[i]]*CSinv[rset[i],rset[i]])
        
    phoS = [0]*nx
    for i in T:
        phoS[i] = math.log(1+gamma[rset[i]]*CSinv[rset[i],rset[i]])
    
    return phoSi, phoS




############## Define Functions for Greedy and Local Search Algorithms ############## 

## Update the inverse matrix by adding a rank-one matrix
def upd_inv_add(X, Xs, opti): 
    Y = 0
    Y = Y + X 
    Y[opti, opti] = Y[opti, opti] + gamma[opti]

    temp =  Xs[:,opti] * Xs[:,opti].T
    Ys = Xs - (gamma[opti]/(1+gamma[opti]*Xs[opti,opti]))* temp    
    return Y, Ys
 	
## Update the inverse matrix by minusing a rank-one matrix
def upd_inv_minus(X, Xs, opti):
    Y = 0 
    Y = Y + X 
    Y[opti, opti] = Y[opti, opti] - gamma[opti]       
    
    temp =  Xs[:,opti] * Xs[:,opti].T
    Ys = Xs + (gamma[opti]/(1-gamma[opti]*Xs[opti,opti]))* temp  
    return Y, Ys

## initial parameters for greedy
def intgreedy(x):
    X = 0.0
    X = X + C
    indexS = np.flatnonzero(x)
    for i in indexS:
        X[i,i] = X[i, i] + gamma[i]
        
    Xs = X.I
    fval = np.linalg.slogdet(X)[1]
    
    return X, Xs, fval


## The update for greedy
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
    
    tempval = 0.0
    tempval = 1-gamma[i]*Xs[i,i]
    if tempval  < 0:
        tempval = 1e-14
        
    val = val + log(tempval) + log(1+gamma[opti]*Ys[opti,opti])
       
    return Y, Ys, opti, val


############## The subgradient used in Frank-Wolfe Algorithm for Continuous M-DDF ############## 
def find_k(x, s, d):
    for i in range(s-1):
        k = i
        mid_val = sum(x[j] for j in range(k+1,d))/(s-k-1)
        if mid_val >= x[k+1]-1e-14 and mid_val < x[k]+1e-14:    
           # print(k)
            return k, mid_val
        
## Compute the subgradient of the continuous relaxation of Resrticted M-DDF
## Compute the objective difference of original M-DDF and resctricted M-DDF (see Proposition 21 in the paper)
## Let subset $S_1$ denote selected points
## Let subset $S_0$ denote discarded points
def grad_fw(S1, S0, fmat, x, s):
    nx = len(x)
    
    val = 0.0
    val = fmat
        
    [a,b] = np.linalg.eigh(fmat) 
    a = a.real # engivalues
    b = b.real # engivectors
    a[a<1e-14] = 0

    sorted_a = sorted(a, reverse=True)     
    k,nu = find_k(sorted_a,s,nx)
    
    # Compute current objective value
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
    temp = V*W*V.T
    val = temp.diagonal()
    val = val.reshape(nx,1)   
    val = list(val)
    
    # compute supgradient of at x
    subg = [0.0]*nx
    for i in range(nx):
        subg[i] = val[i][0,0] # supgradient at x
    
    amu = [0.0]*nx
    for i in S1:
        amu[i] = subg[i]
        
    temp = [0.0]*nx
    for i in range(nx):
        temp[i] = subg[i]
        
    for i in S0:
        temp[i] = min(subg)
    for i in S1:
        temp[i] = min(subg)
        
    temp = np.array(temp)    
    sindex = np.argsort(-temp) 
    
    # solution of linear oracle maximization
    y = [0]*nx
    for i in S1:
        y[i] = 1
    for i in range(s-len(S1)):
        y[sindex[i]] = 1 

    # construct feasible dual solution and dual gap
    nu = subg[sindex[s-len(S1)-1]]
    mu = [0]*nx
    for i in range(s-len(S1)):
        mu[sindex[i]] = subg[sindex[i]]- nu 
        
    dual_gap = (s-len(S1))*nu+sum(mu)+sum(amu)-s
    
    ## Compute the objective difference of original M-DDF and resctricted M-DDF (see Proposition 21 in the paper) 
    fixzero = [0]*nx
    for i in range(nx):
        fixzero[i] = nu + mu[i] - subg[i]
        
    fixone = [0]*nx    
    for i in range(nx):
        fixone[i] = mu[i]
          
    return fval, subg, y, dual_gap, fixzero, fixone     

