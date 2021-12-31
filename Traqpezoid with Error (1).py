# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 09:16:35 2020

@author: emc1977
"""

import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline

es = 0.0001

ansol = 426.6666666666

x = np.linspace(-10,10,100)
y = x**2+4*x-12
plt.plot(x,y)

x0 = -10; x1 = 10;
y0 = x0**2+4*x0-12; y1 = x1**2+4*x1-12;
plt.fill_between([x0,x1],[y0,y1])

plt.xlim([-10,10]); plt.ylim([0,500]);
plt.show()

A = 0.5*(y1 + y0)*(x1 - x0)
#print("Trapezoid area:", A)

def trapz(f,a,b,N=1000):
    
    x = np.linspace(a,b,N+1) # N+1 points make N subintervals
    y = f(x)
    y_right = y[1:] # right endpoints
    y_left = y[:-1] # left endpoints
    dx = (b - a)/N
    T = (dx/2) * np.sum(y_right + y_left)   
    return T

def f(x):
    return x**2+4*x-12

for N in range(2,10,1):
    
    #Ilist = []
        
    integral = trapz(f,-10,10,N)
    
    et = (integral - ansol)/ansol
    
    if abs(et) <= es:
            break
    print(integral, et)
    plt.plot(N,integral, )
    plt.show()

