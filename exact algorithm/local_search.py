#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The implementation of the local search algorithm
"""

import greedy
import datetime
import user


# assign local names to functions in the user and greedy file
findopt =  user.findopt
upd_inv_add = user.upd_inv_add
upd_inv_minus = user.upd_inv_minus
grd = greedy.grd


# Function localsearch needs input n and s; outputs the objectve value, 
# solution, matrix C+\sum_{i in S}gamma[i]*e_i*e_i^T and its inverse, 
# and running time of the local search algorithm

## Let subset $S_1$ denote selected points
## Let subset $S_0$ denote discarded points
def localsearch(S1, S0, n, s):    
    start = datetime.datetime.now()
    
     ## greedy algorithm
    bestf, bestx, X, Xs, gtime = grd(S1, S0, n, s)
    # print("The running time of Greedy algorithm is: ", gtime)
    # print("The output of Greedy algorithm is: ", bestf)


    sel = [i for i in range(n) if bestx[i] == 1] # chosen set
    sel = list(set(sel) - set(S1))
    t = [i for i in range(n) if bestx[i] == 0] # unchosen set  
    t = list(set(t) - set(S0))               
  
    ## local search    
    Y = 0.0
    Ys = 0.0 
    fval = 0.0
    optimal = False

    while(optimal == False):
        optimal = True
        
        for i in sel :
            Y, Ys, index,fval = findopt(X, Xs, i, t, n, bestf)
            if fval > bestf:
                optimal = False                
                bestx[i] = 0
                bestx[index] = 1 # update solution                 
                bestf = fval # update the objective value
                
                X, Xs = upd_inv_add(Y, Ys, index) # update the inverse
                
                sel.remove(i)
                sel.append(index)# update chosen set
                t.remove(index)
                t.append(i)# update the unchosen set
                break

    end = datetime.datetime.now()
    time = (end - start).seconds         
 
    return bestf, bestx, time


