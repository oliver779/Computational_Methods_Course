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
    return S
a = 0
b = 1
n = 100000

f = lambda x: np.exp(-x**2)
solution = simps(f,a,b,n)
print(f"From the simps algorithm we get: {solution}")

c = (a+b)/2
d = (a+c)/2
e = (c+b)/2
 
def integration(f,a,b,c,d,e,n):
    return (b-a)/12*(f(a)+4*f(d)+2*f(c)+4*f(e)+f(b))
print(f"From the adaptive algorithm we get: {integration(f,a,b,c,d,e,n)}")

def error():
    return integration(f,a,b,c,d,e,n) - simps(f,a,b,n)
print(f"The error is:  {error()}")
