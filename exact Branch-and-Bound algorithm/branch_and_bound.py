#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The implementation of Branch-and-Bound algorithms with our proposed
gradient inequalities, submodular inequalities, and optimality cuts
"""

import user 
from gurobipy import *
import numpy as np

# assign local names to functions in the user, local_search, and optcut files
gen_data = user.gen_data
gen_data_mddf = user.gen_data_mddf
rf = user.rf
rhoNi = user.rhoNi
rhoE = user.rhoE
rhoS =  user.rhoS
gradf = user.grad_ineq
  

# S1, S0 denote the cuts (a), (b), respectively
def BranchBound(intx, S1, S0, cut_c, cut_d, cut_e, n, s):

    #Activation of gradient and submodular inequalities
    submod1 = 1
    submod2 = 1
    grad = 1

    # generate data 
    gen_data(n)
    V, intval = gen_data_mddf(n)
    
    temp = list(set(S1).union(set(S0)))  # S_1: set of selected points; S_0: set of discarded points
    rset = list(set(range(n))-set(temp))

    nx = n - len(S1)-len(S0)  # nx: the number of candidates in the restricted DDF
    s = s - len(S1)
    
    rhon = rhoNi(nx, S1, rset, n)
    rhoe = rhoE(nx, S1, rset, n)


    # adds gradient and submodular inequalities
    def lazycuts(m,where):
  
        if where == GRB.callback.MIPSOL:
            y = m.cbGetSolution(xvar)
            yy=np.array([y[i] for i in range(nx)])
            tempinnx = np.argsort(yy)[-s:]
            y = [0]*nx
            for i in tempinnx:
                y[i] = 1
                 
            expr = LinExpr()
            rhosi, rhos = rhoS(y, S1, rset, n)


        # Sumodular Inequality 1:
            if submod1 > 0:
                    expr = wvar
                    rhs = rf(y, S1, rset)
                    for j in range(nx):
                        if y[j] > 0.5:
                            rhs  = rhs - rhon[j]
                            expr = expr - rhon[j]*xvar[j]
                        if y[j] < 0.5:
                            expr = expr - rhos[j]*xvar[j]
                    m.cbLazy(expr,GRB.LESS_EQUAL,rhs)
          

        # Sumodular Inequality 2:
            if submod2 > 0:
                    expr = wvar
                    rhs = rf(y, S1, rset)
                    for j in range(nx):
                        if y[j] > 0.5:
                                rhs  = rhs - rhosi[j]
                                expr = expr - rhosi[j]*xvar[j]
                        if y[j] < 0.5:
                                expr = expr - rhoe[j]*xvar[j]
                    m.cbLazy(expr,GRB.LESS_EQUAL,rhs)
   

        # Gradient Inequalities
            if grad > 0:
                    expr = wvar
                    rhs = rf(y, S1, rset)
                    g, subg = gradf(y, S1, rset, n)
                    for j in range(nx):
                        rhs = rhs - g[j]*y[j]
                        expr = expr - g[j]*xvar[j]
                                
                    m.cbLazy(expr,GRB.LESS_EQUAL,rhs)
                    
                    expr = wvar
                    rhs = rf(y, S1, rset)
                    for j in range(nx):
                        rhs = rhs - subg[j]*y[j]
                        expr = expr - subg[j]*xvar[j]
                       
                    m.cbLazy(expr,GRB.LESS_EQUAL,rhs)
  


 
    def addsubmod1(y):
        expr = LinExpr()
        expr = wvar
        yy = np.array([y[i] for i in range(nx)])
        tempinnx = np.argsort(yy)[-s:]
        
        y = [0]*nx
        for i in tempinnx:
            y[i] = 1
            
        rhs = rf(y, S1, rset)
        rhosi, rhos = rhoS(y, S1, rset, n)
        for j in range(nx):		
            if y[j] > 0.5:
                rhs  = rhs - rhon[j]
                expr = expr - rhon[j]*xvar[j]
            if y[j] < 0.5:
                expr = expr - rhos[j]*xvar[j]

        m.addConstr(expr, GRB.LESS_EQUAL, rhs)
        m.update()


    def addsubmod2(y):
        expr = LinExpr()
        expr = wvar
        yy=np.array([y[i] for i in range(nx)])
        tempinnx = np.argsort(yy)[-s:]
        
        y = [0]*nx
        for i in tempinnx:
            y[i] = 1
            
        rhs = rf(y, S1, rset)
        rhosi, rhos = rhoS(y, S1, rset, n)
        for j in range(nx):
                if y[j] > 0.5:
                        rhs  = rhs - rhosi[j]
                        expr = expr - rhosi[j]*xvar[j]
                if y[j] < 0.5:
                        expr = expr - rhoe[j]*xvar[j]
                        
        m.addConstr(expr, GRB.LESS_EQUAL, rhs)
        m.update()

    
    def addgrad(y):
        expr = LinExpr()
        expr = wvar
        yy=np.array([y[i] for i in range(nx)])
        tempinnx = np.argsort(yy)[-s:]
        
        y = [0]*nx
        for i in tempinnx:
            y[i] = 1
            
        rhs = rf(y, S1, rset)
        g, subg = gradf(y, S1, rset, n)
        
        for j in range(nx):
            rhs = rhs - g[j]*y[j]
            expr = expr - g[j]*xvar[j]
        m.addConstr(expr, GRB.LESS_EQUAL, rhs)
        m.update()
        
        expr = LinExpr()
        expr = wvar
        rhs = rf(y, S1, rset)
        for j in range(nx):
            rhs = rhs - subg[j]*y[j]
            expr = expr - subg[j]*xvar[j]
        m.addConstr(expr, GRB.LESS_EQUAL, rhs)
        m.update()
        

    #create model
    m = Model()

    # add variables
    xvar = {}
    for i in range(nx):
        xvar[i] = m.addVar(obj=0.0, vtype=GRB.BINARY, name='x'+str(i))
                       
    wvar = m.addVar(obj=1.0, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='w')
    m.update()
    
    zvar = {}
    for i in range(len(cut_e)):
            zvar[i] = m.addVar(obj=0.0, vtype=GRB.BINARY, name='z'+str(i))
    
    for i in range(len(cut_e)):
        m.addConstr(xvar[rset.index(cut_e[i][1])] >= zvar[i])
        m.update()
        m.addConstr(zvar[i] >= xvar[rset.index(cut_e[i][0])])
        m.update()
        
    for i in range(len(cut_c)):
        m.addConstr(xvar[rset.index(cut_c[i][0])] + xvar[rset.index(cut_c[i][1])] >= 1)
        m.update()
        
    for i in range(len(cut_d)):
        m.addConstr(xvar[rset.index(cut_d[i][0])] + xvar[rset.index(cut_d[i][1])] <= 1)
        m.update()

    # add cardinality constraint
    m.addConstr(quicksum(xvar[i] for i in range(nx)), GRB.EQUAL, s, 'card')
    m.update()   

    # specify optimization objective
    m.modelSense = GRB.MAXIMIZE
    m.update()


    addsubmod1(intx)
    addsubmod2(intx)
    addgrad(intx)
    

    MIPGap=1e-10
    m.params.LazyConstraints = 1
    m.params.OutputFlag = 1
    m.params.timelimit = 3600*4 # time limit is four hours
    m.optimize(lazycuts)
   

    # get status, upper and lower bounds
    status = m.status
    UB = m.ObjBound
    LB = m.ObjVal
    
    xsol=[0]*nx  # the output solution of B&B at optimality or reached the time limit
    for i in range(nx):
        xsol[i] = xvar[i].x

    obj_val = rf(xsol, S1, rset)  # the objective value of DDF at the output solution
    w = wvar.x # the output value of B&B
    
    return w, obj_val, UB, LB
   

