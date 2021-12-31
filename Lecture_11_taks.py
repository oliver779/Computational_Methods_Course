# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 17:55:16 2021

@author: Oliver
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
"""Numerical Integration Methods"""

"""Rectangular Rule""" """INTEGRATION"""

def calculate_dx (a, b, n):
    return (b-a)/float(n)

def rect_rule (f, a, b, n):
    total = 0.0
    dx = calculate_dx(a, b, n)
    for k in range (0, n):
        total = total + f((a + (k*dx)))
    return dx*total

def f(x):
    return x**2
print(rect_rule(f, 0, 10, 100000))

"""Trapezoidal Rule"""

def trapz(f,a,b,N=50): 
    x = np.linspace(a,b,N+1) # N+1 points make N subintervals
    y = f(x)
    y_right = y[1:] # right endpoints
    y_left = y[:-1] # left endpoints
    dx = (b - a)/N
    T = (dx/2) * np.sum(y_right + y_left)
    return T

def f(x):
    return np.exp(-x**2)

a=1#float(input("Please enter a value for the lower bound:\n"))
b=-1#float(input("Please enter a value for the upper bound:\n"))
n=1000#int(input("Please enter a value for the number of intervals:\n"))
print(trapz(f,a,b,n)) 

# Graphing the Trapezoidal Rule


x = np.linspace(-1.5,1.5,100)
y = np.exp(x**2)
plt.plot(x,y)
x0 = 0; x1 = 1;
y0 = np.exp(x0**2); y1 = np.exp(x1**2);
plt.fill_between([x0,x1],[y0,y1])
plt.xlim([-1.5,1.5]); plt.ylim([0,10]);
plt.show()
A = 0.5*(y1 + y0)*(x1 - x0)
print("Trapezoid area:", A)


"""Simpsons Rule Method"""

def simps(f,a,b,N=50):
    if N % 2 == 1:
        raise ValueError("N must be an even integer.")
    dx = (b-a)/N
    x = np.linspace(a,b,N+1)
    y = f(x)
    S = dx/3 * np.sum(y[0:-1:2] + 4*y[1::2] + y[2::2]) # x[startAt:endBefore:skip]
    return S
# https://stackoverflow.com/questions/9027862/what-does-listxy-do
f = lambda x: x**3
solution = simps(f,1,2,24)
print(solution)

"""Adaptive Simpson Algorithm"""

es = 0.0001

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

ansol = scipy.integrate.quad(f,-10,10)

for N in range(2,10,1):
    es = 0.0001
    integral = simps(f,-10,10,N)
    et = (integral - ansol[0])/ansol[0]
    if abs(et) <= es:
        print(f'integral and error {integral, et}')
        break

# IDK TRY TOMORROW WHEN YOU ARE LESS TIRED


# "structured" adaptive version, translated from Racket
def _quad_simpsons_mem(f, a, fa, b, fb):
    """Evaluates the Simpson's Rule, also returning m and f(m) to reuse"""
    m = (a + b) / 2
    fm = f(m)
    return (m, fm, abs(b - a) / 6 * (fa + 4 * fm + fb))

def _quad_asr(f, a, fa, b, fb, eps, whole, m, fm):
    """
    Efficient recursive implementation of adaptive Simpson's rule.
    Function values at the start, middle, end of the intervals are retained.
    """
    lm, flm, left  = _quad_simpsons_mem(f, a, fa, m, fm)
    rm, frm, right = _quad_simpsons_mem(f, m, fm, b, fb)
    delta = left + right - whole
    if abs(delta) <= 15 * eps:
        return left + right + delta / 15
    return _quad_asr(f, a, fa, m, fm, eps/2, left , lm, flm) +\
           _quad_asr(f, m, fm, b, fb, eps/2, right, rm, frm)

def quad_asr(f, a, b, eps):
    """Integrate f from a to b using Adaptive Simpson's Rule with max error of eps."""
    fa, fb = f(a), f(b)
    m, fm, whole = _quad_simpsons_mem(f, a, fa, b, fb)
    return _quad_asr(f, a, fa, b, fb, eps, whole, m, fm)

from math import sin
print(quad_asr(sin, 0, 1, 1e-09))




































