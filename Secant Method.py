# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 14:24:24 2021

@author: Oscar
"""

'''Approximate solution of f(x)=0 on interval [a,b] by the secant method.

Parameters
----------
f : function
    The function for which we are trying to approximate a solution f(x)=0.
a,b : numbers
    The interval in which to search for a solution. The function returns
    None if f(a)*f(b) >= 0 since a solution is not guaranteed.
N : (positive) integer
    The number of iterations to implement.

Returns
-------
m_N : number
    The x intercept of the secant line on the the Nth interval
        m_n = a_n - f(a_n)*(b_n - a_n)/(f(b_n) - f(a_n))
    The initial interval [a_0,b_0] is given by [a,b]. If f(m_n) == 0
    for some intercept m_n then the function returns this solution.
    If all signs of values f(a_n), f(b_n) and f(m_n) are the same at any
    iterations, the secant method fails and return None.
'''
import math
# Resembles the Closed Interval Method
def secant(f, a,b ,N):
    
    if f(a)*f(b) >=0:
        print("Secant method fails.")
        return None
    
    for i in range(1,N+1):
        x_n = a - f(a)*(a-b)/(f(a)-f(b))
        if f(a)*f(x_n) < 0:
            a = a
            b = x_n
        elif f(b)*f(x_n) < 0:
            a = x_n
            b = b
        elif f(x_n) == 0:
            print("Found exact solution.")
            return x_n, i
        else:
            print("Secant method fails.")
            return None
    print('Found solution after', i, 'iterations.')
    return a - f(a)*(a-b)/(f(a)-f(b))

f = lambda x: x**2 +4*x -12
solution = secant(f,-5,5,25)
print(solution)
abs_error = (solution - 2)/2
print(abs_error)
