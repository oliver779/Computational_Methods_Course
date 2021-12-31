# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 22:41:21 2021

@author: Oliver
"""
import numpy as np
import matplotlib.pyplot as plt
import timeit


x = np.linspace(-10,10,100)
y = x**2+4*x-12
plt.plot(x,y,'r')

plt.show
func = lambda x: x**2+4*x-12

def bisection(a,b): 
    if (func(a) * func(b) >= 0): 
            print("You have not assumed right a and b\n") 
            return 
    c = a 
    while ((b-a) >= 0.01):
        # Find middle point
        c = (a+b)/2
        # Check if middle point is root 
        if (func(c) == 0.0): 
            break
        # Decide the side to repeat the steps 
        if (func(c)*func(a) < 0): 
            b = c 
        else: 
            a = c 
        print("The value of root is : ","%.4f"%c) 
# Driver code 
# Initial values assumed 
a = int(input("Please give the value of a: "))
b = int(input("Please give the value of b: "))
start = timeit.default_timer()
bisection(a, b)
stop = timeit.default_timer()
print('Time of original bisection: ', stop - start)  
