# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 23:04:55 2021

@author: Oliver
"""

import math
import numpy as np
import timeit
def f(x):
    f = math.pow(x,2) + 4*x - 12 -math.pow(x, 3)
    return f

def regulaFalsi(a,b,TOL,N):
    i = 1
    FA = f(a)
    print("%-20s %-20s %-20s %-20s %-20s" % ("n","a_n","b_n","p_n","f(p_n)"))
    while(i <= N):
            p = (a*f(b)-b*f(a))/(f(b) - f(a))
            FP = f(p)
            if(FP == 0 or np.abs(f(p)) < TOL):
                break
            else:
                print("%-20.8g %-20.8g %-20.8g %-20.8g %-20.8g\n" % (i, a, b, p, f(p)))
            i = i + 1
            if(FA*FP > 0):
                b = p
            else:
                a = p
    return
    print(i)
start = timeit.default_timer()
regulaFalsi(-1, 10, 0.0001, 100)
stop = timeit.default_timer()
print('Time of changed: ', stop - start)  

def regulaFalsi2(a,b,TOL,N):
    i = 1
    FA = f(a)
    print("%-20s %-20s %-20s %-20s %-20s" % ("n","a_n","b_n","p_n","f(p_n)"))
    while(i <= N):
            p = (a*f(b)-b*f(a))/(f(b) - f(a))
            FP = f(p)
            if(FP == 0 or np.abs(f(p)) < TOL):
                break
            else:
                print("%-20.8g %-20.8g %-20.8g %-20.8g %-20.8g\n" % (i, a, b, p, f(p)))
            i = i + 1
            if(FA*FP > 0):
                a = p
            else:
                b = p
    return
    print(i)

start2 = timeit.default_timer()
regulaFalsi2(1, 10, 0.0001, 100)
stop2 = timeit.default_timer()
print('Time of original false position: ', stop2 - start2) 