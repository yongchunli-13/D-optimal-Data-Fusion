#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The derivation of optimality cuts under five different settings: 
(a) (b) (c) (d) (e) as defined in subsection 5.2 in the paper 
"""


import numpy as np
import local_search
import frank_wolfe
import datetime

# assign local names to functions in the local_search and frank_wolfe files
localsearch = local_search.localsearch
frankwolfe = frank_wolfe.frankwolfe
resfrankwolfe = frank_wolfe.res_frankwolfe


def optcut(n, s):
    start = datetime.datetime.now()
    fval, xsol, time = localsearch([], [], n, s)
    S1, S0, fixone, fixzero, init_cut_gap, cxsol = frankwolfe(fval, xsol, n, s) 
     
    #### derive cut (b) ####
    indexS = []
    indexT = S0
    temp = np.array(fixzero)
    sortinx = np.argsort(-temp)
    setzero = list(set(range(n))-set(S0))
    tempinx = np.argsort(cxsol)[-2:]
    cutgap = -1
    for i in range(n):
        if temp[sortinx[i]] < init_cut_gap and sortinx[i] in setzero:
            indexS= []
            indexS = S1
            indexS.append(sortinx[i]) # suppose select i-th point 
            
            x = [0]*n
            for j in range(n):
                x[j] = cxsol[j]
                
            x[sortinx[i]] =  1.0    
            x[tempinx[0]] = x[tempinx[0]]-(1.0- cxsol[sortinx[i]])/2
            x[tempinx[1]] = x[tempinx[1]]-(1.0- cxsol[sortinx[i]])/2
            
            cutgap = resfrankwolfe(indexS, indexT, fval, x, n, s, 1e-5)
    
            if cutgap < 0: # restricted DDF < DDF if i-th point is selected; Hence, discard i-th point
                S0.append(sortinx[i])
            indexS.remove(sortinx[i])
            
        if cutgap > 1e-2:
            break    
        
    #### derive cut (a) ### 
    indexS = []
    indexT = []

    sone = [i for i in range(n) if cxsol[i] > 0.5]
    sone = list(set(sone)-set(S1))
    mininx = cxsol.index(0)
    cutgap = -1
    for i in sone:
        indexT = []
        indexT = S0
        indexT.append(i) # suppose discard i-th point 
        
        x = [0]*n
        for j in range(n):
            x[j] = cxsol[j]
            
        x[i] =  0.0    
        x[mininx] = cxsol[i]
        
        cutgap = resfrankwolfe(indexS, indexT, fval, x, n, s, 1e-5)

        if cutgap < 0: # restricted DDF < DDF if i-th point is discarded; Hence, select i-th point
            S1.append(i)
            
        indexT.remove(i)

            
    #### derive cut (c) #### 
    indexS = []
    indexT = [] 
    S = [i for i in range(n) if fixone[i] > 1e-3 or cxsol[i] > 0.85] 
    S = list(set(S)-set(S1))
    Setone = []  
    cutgap = -1
    tempinx = np.argsort(-np.array(cxsol))[-2:]
    for i in range(len(S)):
        for j in range(i+1, len(S)):       
            indexT = []
            indexT = [S[i], S[j]] # suppose that i-th and j-th points are discarded
                  
            x = [0]*n
            for l in range(n):
                x[l] = cxsol[l]
                   
            x[tempinx[0]] = cxsol[S[i]]
            x[tempinx[1]] = cxsol[S[j]]
            x[S[i]] = 0.0
            x[S[j]] = 0.0
            
            cutgap = resfrankwolfe(indexS, indexT, fval, x, n, s, 1e-4)
            
            if cutgap < 0: # restricted DDF < DDF; Hence, select at least one of i-th and j-th points
                Setone.append(indexT)  
            if cutgap > 1e-2:
                break 
 

                   
    #### derive cut (d) ####  
    if len(S0)>0:
        S = [i for i in range(n) if fixzero[i] > 0 and fixzero[i] < fixzero[S0[-1]]] 
    else:
        S = [i for i in range(n) if fixzero[i] > 0] 
    S = list(set(S)-set(S0))    
    Setzero = []  
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            # suppose that i-th and j-th points are selected
            tempS = []
            if fixzero[S[i]] + fixzero[S[j]] > init_cut_gap: # restricted DDF < DDF; Hence, discard at least one of i-th and j-th points
                tempS = [S[i], S[j]]
                Setzero.append(tempS) 
            if j > i+2:
                break            
     
            
     
    #### derive cut (e) #####
    indexS = []
    indexT = []
    temp = np.array(fixone)
    sortinx = np.argsort(-temp)
    mininx = cxsol.index(0)
    tempinx = np.argsort(cxsol)[-2:]
    cutgap = 1
    szero = [i for i in range(n) if fixzero[i] > 1e-2]
    sone = [i for i in range(n) if fixone[i] > 1e-4]
    szero = list(set(szero)-set(S0))
    sone = list(set(sone)-set(S1))
    szero = list(set(szero)-set(tempinx))
    szero = list(set(szero)-set([mininx]))
    sone = list(set(sone)-set(tempinx))
    sone = list(set(sone)-set([mininx]))
    Setjunc = []
    t = 0
    for i in szero:
        t = t+1
        for j in sone:
            if i !=j :
                indexS, indexT = [], []
                indexS = [i] # suppose that i-th point is selected and the j-th point is discarded
                indexT = [j]
                
                x = [0]*n
                for l in range(n):
                    x[l] = cxsol[l]
                    
                x[j] =  0.0    
                x[mininx] = cxsol[j]
                
                x[i] =  1.0    
                x[tempinx[0]] = x[tempinx[0]]-(1.0- cxsol[i])/2
                x[tempinx[1]] = x[tempinx[1]]-(1.0- cxsol[i])/2
                
                
                cutgap = resfrankwolfe(indexS, indexT, fval, x, n, s, 1e-4)
        
                if cutgap < 0: # restricted DDF < DDF; Hence, obtain cut (e) about points (i, j)
                    Setjunc.append([i,j])
        if t>5:
            break
             
       
    
    print("The number of optimality cut (a) is", len(S1))
    print("The number of optimality cut (b) is", len(S0))
    print("The number of optimality cut (c) is", len(Setone))
    print("The number of optimality cut (d) is", len(Setzero))
    print("The number of optimality cut (e) is", len(Setjunc))
    
    end = datetime.datetime.now()
    time = (end-start).seconds
    
    return S1, S0, Setone, Setzero, Setjunc, time


    