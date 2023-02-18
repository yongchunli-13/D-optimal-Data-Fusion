#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The implementation of the Frank-Wolfe algorithm 
for solve the continuous relaxation of M-DDF, i.e., $\hat{z}_M$ 

Besides, compute the gap between the continuous relaxation of restricted M-DDF 
and the lower bound of DDF to derive optimality cuts
"""


import user
import numpy as np

# assign local names to functions in the user file
gen_data = user.gen_data
gen_data_mddf = user.gen_data_mddf
grad = user.grad_fw

# Function frankwolfe needs input n and s; outputs the solution, 
# continuous relaxation value $\hat{z}_M$ and running time of the Frank-Wolfe algorithm
# In addition, derive the straightforward optimality cuts
def frankwolfe(LB, x, n, s): 
    
    
    gen_data(n)
    V, intval = gen_data_mddf(n)
        
    gamma_t = 0.0  
    t = 0.0
    mindual = 1e+10
    dual_gap = 1 # duality gap
    Obj_f = 1 # primal value
    alpha = 1e-5 # target accuracy

    cut_gap = 0.0 # the gap of DDF and restricted DDF to derive optimality cuts    
    xsol = [2]*n
    S1 = []  # store selected points
    S0 = []  # store discarded points    
    fixzero = [0]*n
    fixone = [0]*n
            
    sel = np.flatnonzero(x) 
    fmat = 0.0
    for i in sel:
        fmat = fmat + x[i]*V[i].T*V[i]
        
    while(dual_gap/abs(mindual) > alpha or t<=10):
        Obj_f, subgrad, y, dual_gap, fixzero, fixone = grad([], [], fmat, x, s)
        
        t = t + 1
        gamma_t = 2/(t+2) # step size
        
        x = [(1-gamma_t)*x_i for x_i in x] 
        y = [gamma_t*y_i for y_i in y]
        
        fmat = (1-gamma_t) * fmat
        sel = np.flatnonzero(y) 
           
        ymat = 0.0
        for i in sel:
            ymat = ymat + V[i].T*V[i]
            
        fmat = fmat + gamma_t*ymat
        
        x = np.add(x,y).tolist() # update the current solution x
        
        mindual = min(mindual, Obj_f+dual_gap) # update the upper bound
        
        ## derive optimality cuts
        cut_gap =  Obj_f + dual_gap  - LB
        for i in range(n):
            if cut_gap < fixzero[i]:  # restricted DDF < DDF if i-th point is selected; Hence, discard i-th point                
                xsol[i] = 0
            if cut_gap < fixone[i]:  # restricted DDF < DDF if i-th point is discarded; Hence, select i-th point                 
                xsol[i] = 1
       
        S0 = [i for i in range(n) if xsol[i] == 0] # discarded points
        S1 = [i for i in range(n) if xsol[i] == 1] # selected points
        
    
    return S1, S0, fixone, fixzero, cut_gap, x


# Frank-wolfe for the continuous relaxation of restricted M-DDF
# Suppose that subsets $S_1, S_0$ denote selected and discarded points, respectively
def res_frankwolfe(S1, S0, LB, x, n, s, talpha): 
    
    gen_data(n)
    V, intval = gen_data_mddf(n)
        
    gamma_t = 0.0  
    t = 0.0
    mindual = 1e+10
    dual_gap = 1 # duality gap
    Obj_f = 1 # primal value
    alpha = 1e-4 # accuracy
       
    sel = np.flatnonzero(x) 
    fmat = 0.0
    for i in sel:
        fmat = fmat + x[i]*V[i].T*V[i]
    
    cutgap = 0.0 # the gap of DDF and restricted DDF to derive optimality cuts    
    while(dual_gap/abs(Obj_f) > alpha and cutgap >= 0):
        Obj_f, subgrad, y, dual_gap, fixzero, fixone = grad(S1, S0, fmat, x, s)
        
        t = t + 1
        gamma_t = 2/(t+2) # step size
        
        x = [(1-gamma_t)*x_i for x_i in x] 
        y = [gamma_t*y_i for y_i in y]
        
        fmat = (1-gamma_t) * fmat
        sel = np.flatnonzero(y) 
           
        ymat = 0.0
        for i in sel:
            ymat = ymat + V[i].T*V[i]
            
        fmat = fmat + gamma_t*ymat
        
        x = np.add(x,y).tolist() # update x
        
        mindual = min(mindual, Obj_f+dual_gap) # update the upper bound
    
        cutgap =  mindual  - LB # upper bound of Restricted DDF - lower bound of DDF
        

    
    if dual_gap > cutgap + 1e-4 and talpha < 1e-4:  #talpha: target accuracy
        alpha = 1e-4
        while(dual_gap/abs(Obj_f+intval) > alpha and cutgap > 0):
            Obj_f, subgrad, y, dual_gap, fixzero, fixone = grad(S1, S0, fmat, x, s)
            
            t = t + 1
            gamma_t = 2/(t+2) # step size
            
            x = [(1-gamma_t)*x_i for x_i in x] 
            y = [gamma_t*y_i for y_i in y]
            
            fmat = (1-gamma_t) * fmat
            sel = np.flatnonzero(y) 
               
            ymat = 0.0
            for i in sel:
                ymat = ymat + V[i].T*V[i]
                
            fmat = fmat + gamma_t*ymat
            
            x = np.add(x,y).tolist() # update x
            
            mindual = min(mindual, Obj_f+dual_gap) # update the upper bound
        
            cutgap =  mindual - LB # upper bound of Restricted DDF - lower bound of DDF
            
    return cutgap



