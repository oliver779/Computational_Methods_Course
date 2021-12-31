# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 18:57:23 2021

@author: Oliver
"""
# Ctrl + 1 = #
# def poly_horner(x, coeff):
#     result = coeff[-1]
#     for i in range(-2, -len(coeff)-1, -1):
#         result = result*x + coeff[i]
#     return result

# f =lambda x: x**5 +2*x
# coeff = [5,1]
# poly_horner(f, coeff)

# Python program for
# implementation of Horner Method
# for Polynomial Evaluation
 
# returns value of poly[0]x(n-1)
# + poly[1]x(n-2) + .. + poly[n-1]
def horner(poly, n, x):
    # Initialize result
    result = poly[0] 
    # Evaluate value of polynomial
    # using Horner's method
    for i in range(1, n):
        result = result*x + poly[i]
    return result
  
# Driver program to
# test above function.
x = 3
# Let us evaluate value of
f = 2*x**3 - 6*x**2 + 2*x - 1 
poly = [2, -6, 2, -1, 2]
n = len(poly)
 
print("Value of polynomial is " , horner(poly, n, x))
 
# This code is contributed
# by Anant Agarwal.















