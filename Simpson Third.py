# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 10:13:14 2020

@author: chris
"""
import numpy as np

def simps(f,a,b,N=50):
    if N % 2 == 1:
        raise ValueError("N must bean even integer.")
    dx = (b-a)/N
    x = np.linspace(a,b,N+1)
    y = f(x)
    
    S = dx/3 * np.sum(y[0:-1:2] +4*y[1::2] + y[2::2])
    return S

#f = lambda x: x**2+4*x-12
#solution = simps(f,-10,10,2)
#print(solution)