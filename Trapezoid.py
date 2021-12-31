# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 09:53:04 2020

@author: chris
"""
import numpy as np


def trapz(f,a,b,N=50):
    x = np.linspace(a,b,N+1) # N+1 points make N subintervals
    y = f(x)
    y_right = y[1:] # right endpoints
    y_left = y[:-1] # left endpoints
    dx = (b - a)/N
    T = (dx/2) * np.sum(y_right + y_left)
    return T

#def f(x):
#    return x**2+4*x-12

#a=-10
#b=10
#n=10000
#print(trapz(f,a,b))