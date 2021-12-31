# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 21:39:53 2021

@author: Oliver
"""
import matplotlib.pyplot as plt
import numpy as np
import timeit

x=np.linspace(-10,10,50)
f =  x**2 + 4*x - 12
function = lambda x: x**2 + 4*x - 12
plt.plot(x,f,'g')
input_a =  int(input("Please give value of a: "))
input_b =  int(input("Please give value of b: "))

def bisection(f,a,b,N):
    if function(a)*function(b) >= 0:
        print("Bisection method fails.")
        return None
    a_n = a
    b_n = b
    for n in range(1,N+1):
        m_n = (a_n + b_n)/2
        f_m_n = function(m_n)
        if function(a_n)*f_m_n < 0:
                a_n = a_n
                b_n = m_n
        elif function(b_n)*f_m_n < 0:
                a_n = m_n
                b_n = b_n
        elif f_m_n == 0:
            print(f"Found exact solution after {n} tries")
            return m_n
        else:
            print("Bisection method fails.")
            return None
    return (a_n + b_n)/2
    
start = timeit.default_timer()
approx_phi = bisection(f,input_a,input_b,25)
stop = timeit.default_timer()
print('Time of original bisection without  error: ', stop - start) 
print(f'The zero is at: {approx_phi}')
