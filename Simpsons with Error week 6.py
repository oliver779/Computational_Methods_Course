# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 09:23:27 2020

@author: emc1977
"""

import numpy as np
import matplotlib.pyplot as plt

es = 0.0001

ansol = 426.666666666666666666

def simps(f,a,b,N):
    
    if N % 2 == 1:
        raise ValueError("N must be an even integer.")
    dx = (b-a)/N
    x = np.linspace(a,b,N+1)
    y = f(x)
    S = dx/3 * np.sum(y[0:-1:2] + 4*y[1::2] + y[2::2])
    #print(S)
    return S

def f(x):
    return x**2+4*x-12

for N in range(2,10,1):
    
    es = 0.0001

    ansol = 426.666666666666666666

    integral = simps(f,-10,10,N)
    
    et = (integral - ansol)/ansol
    
    if abs(et) <= es:
            break
    print(integral, et)
    
print(integral, et)

