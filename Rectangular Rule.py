# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 09:36:12 2020

@author: chris
"""

def rect_rule (f, a, b, n):
    total = 0.0
    dx = (b-a)/float(n)
    for k in range (0, n):
        total += f((a +(k*dx)+0.5*dx))
    return dx*total

#def f(x):
#    return x**2+4*x-12

#print(rect_rule(f, -10, 10, 10000))