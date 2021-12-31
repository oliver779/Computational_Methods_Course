# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 21:42:30 2020

@author: emc1977
"""

import numpy as np
import matplotlib.pyplot as plt

es = 0.0001

ansol = 426.6666666666

def calculate_dx (a, b, n):
	return (b-a)/float(n)

def rect_rule (f, a, b, n):
	total = 0.0
	dx = calculate_dx(a, b, n)
	for k in range (0, n):
        	total = total + f((a + (k*dx)))
	return dx*total

def f(x):
    return x**2+4*x-12

for n in range(2,10,1):
    
    Ilist = []

    integral = rect_rule(f, -10, 10, n)
    
    Ilist.append(integral)
    
    et = (integral - ansol)/ansol
    
    if abs(et) <= es:
            break
    print(integral, et)
    
    plt.plot(n,integral)
    plt.show()
    
    


