# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 10:34:33 2020

@author: chris
"""
import numpy as np
#def func(x):
#    return x**2+4*x-12
# Function to perform calculations
def calculate(func,lower_limit, upper_limit, interval_limit ):
    interval_size = (upper_limit - lower_limit) / interval_limit
    sum = func(lower_limit) + func(upper_limit);
# Calculates value till integral limit
    for i in range(1, interval_limit ):
        if (i % 3 == 0):
            sum = sum + 2 * func(lower_limit + i * interval_size)
        else:
            sum = sum + 3 * func(lower_limit + i * interval_size)
    return ((float( 3 * interval_size) / 8 ) * sum )
    
# driver function

#interval_limit = 30

#lower_limit = -10
#upper_limit = 10
#integral_res = calculate(func,lower_limit, upper_limit, interval_limit)
# rounding the final answer to 6 decimal places
#print (round(integral_res, 6))