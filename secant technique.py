# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 01:10:20 2021

@author: Oliver
"""
import timeit

def secant(f,a,b,N): 
    if f(a)*f(b) >= 0: 
        print("Secant method fails.") 
        return None 
    a_n = a
    b_n = b 
    for n in range(1,N+1): 
        m_n = a_n - f(a_n)*(b_n - a_n)/(f(b_n) - f(a_n)) 
        f_m_n = f(m_n) 
        if f(a_n)*f_m_n < 0: 
            a_n = a_n 
            b_n = m_n
        elif f(b_n)*f_m_n < 0: 
            a_n = m_n 
            b_n = b_n
        elif f_m_n == 0: 
            print("Found exact solution.") 
            return m_n, n
        else:
            print("Secant method fails.") 
            return None 
    return a_n - f(a_n)*(b_n - a_n)/(f(b_n) - f(a_n))

fun = lambda x: x**2 + 4*x - 12
a = 1
b = 10
n = 100

start = timeit.default_timer()
solution = secant(fun,a,b,n)
stop = timeit.default_timer()
print('Time of original secant: ', stop - start) 
print(solution)