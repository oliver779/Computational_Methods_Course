# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 16:37:00 2021

@author: Oliver
"""

# calculate the integration step interval dx using the 
# defined bounds a and b and the number of intervals chosen

def calculate_dx(a,b,n):
    return (b-a)/float(n)

# Implement the rectangle rule calculation of integral based 
# on calculation of area in each rectangular strip.

def rect_rule(f,a,b,n):
    total = 0.0
    dx = calculate_dx(a,b,n)
    for k in range(0,n):
        total += f((a+(k*dx)))
    return dx*total

# The function to be integrated

def f(x):
    return x**2

print(rect_rule(f,0,10,100000))