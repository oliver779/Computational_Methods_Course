# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 21:41:25 2021

@author: Oscar
"""

import math
import numpy as np

def regularFalsi(f, a, b, Tol, N):
    i = 1
    fa = f(a)
    fb = f(b)
    print("%-20s %-20s %-20s %-20s %-20s" % 
          ("n","a_n","b_n","p_n","f(p_n)"))
    while(i <= N):
        sol = (a*f(b)-b*f(a))/(f(b)-f(a))
        fp = f(sol)
        if(fp==0 or np.abs(f(sol))<Tol):
            break
        else:
            print("%-20.8g %-20.8g %-20.8g %-20.8g %-20.8g\n" % (i, a, b, sol, f(sol)))
        i = i + 1
        if(fa*fb > 0):
            a = sol
        else:
            b = sol
    abs_error=(sol-2)/2
    print(abs_error)
    return sol
    print (i)

sol = 0
a =-1
b = 5
Tol = 1E-6
N = 1000
f = lambda x: x**2+4*x-12
print("Sample input: regulaFalsi(f, 1, 2, 10**-4, 100)")
approxi_phi = regularFalsi(f, a, b, Tol, N)
print(approxi_phi)
