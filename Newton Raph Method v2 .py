# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 13:37:44 2021

@author: Oscar
"""

# Newton Raph Method is anohter open root method to finding the roots  of a function
    # The function in question is evaluated at an initially choosen point
    # A tangent line wrt the point is then drawn and intersects the x-axis
    # Repeats until the root is found
        # x_new =  x - f(x)/f'(x)
import sympy as sp
        
# Define The required Parameters 
x0 = 1 #int(input("Initial Estimate:"))
N = 100 #int(input("Max Iterations:"))
e = 0.000001#float(input("Tolerance:"))
# Equation that needs to be evaluated

f = lambda x: x**2 +4*x -12
df = lambda x: 2*x +4

def NewtonRapshon(f, df, e, N):
    
    xn = x0
    
    for i in range(0,N):
        # If the value of the function at x is less than the tolerance, we print xn as it is close to the root
        if abs(f(xn)) < e:
            abs_error=(xn-2)/2
            print('Found solution after',i,'iterations.')
            return xn, abs_error
        # Find the derivative at the value of xn, if none exists
        if df(xn) == 0:
            print('Zero derivative. No solution found.')
            return None
        # Calculate the new value of xn and repeat the process
        xn = xn - f(xn)/df(xn)
    
    # If the Max number of iterations are reached, stop the program  
    print('Exceeded maximum iterations. No solution found.')

    return 

solution = NewtonRapshon(f, df, e, N)
print(solution)