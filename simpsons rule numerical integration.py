# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 17:00:37 2021

@author: Oliver
"""
import numpy as np

# Define the function, limits and the number of evaluation strips N

def simps(f,a,b,n):
    if n % 2 == 1:
        raise ValueError("N must be an integer.")
    dx = (b-a)/n
    x = np.linspace(a,b,n+1)
    y = f(x)
    S = dx/3 *np.sum(y[0:-1:2] + 4*y[1::2] + y[2::2])
    print(S)
    return S

f = lambda x: x**3
solution = simps(f,1,2,24)
print(solution)