# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 16:46:16 2021

@author: Oliver
"""

# Define the core Trapezoid rule function defining the function, a,b 
# and the number of strips N used. T is the integral in [a,b].
import numpy as np
import matplotlib.pyplot as plt

def trapz(f,a,b,N):
    x = np.linspace(a,b,N+1) # N+1 points make N subintervals
    y = f(x)
    y_right = y[1:] # right endpoints
    y_left = y[:1] # left endpoints
    dx = (b-a)/N
    T = (dx/2)* np.sum(y_right + y_left)
    return T

# Define the function to be integrated

def f(x):
    return np.exp(-x**2)

a = 1
b = 10
n = 2000000
# N =50

print(trapz(f,a,b,n))

# graphing the trapezoidal rule
# defining x and y and plotting the curve
x = np.linspace(-1.5,1.5,100)
y = np.exp(x**2)
plt.plot(x,y)

# these commands fill a defined region of the graph to form a trapezoid area. then it calculates 
#the area,A, shown within the trapezoid geometrically. this A is the approximation fo the integral oif 
#the function
x0 = 0
x1 = 1
y0 = np.exp(x0**2)
y1 = np.exp(x1**2)
plt.fill_between([x0,x1],[y0,y1])
plt.show()
A = 0.5*(y1+y0)*(x1 - x0)
print("trapezoidal area", A)

