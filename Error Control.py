# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 09:24:00 2021

@author: Oscar
"""

# True Error, E_t = True Value - imperfect value [Absolute Error = True - Measured]
# Fractional Relative Error E_f= True Error/ True Value [Relative Error = AbsE/Value ]
# Approx Fractional Error E_a = (current approx - previous approx)/current approx
    #for |E_a| to be acceptable, E_a < E_s: E_s is a pre-selected acceptable error
        #In practice?
            #Define Es
            #Choose function of interest f(x)
            #Choose a value of x to evalutate f(x)
            #Express function in form for numerical calculation: ie a Series
            #Calc with the number of series terms required to satisfy Es
            
# Evaluate e^0.5 numerically
"""
import math

x = 0.5 # define x to be 1/2
e_to_x = 0 

for i in range(6): # For loop and the +- operator to define the Exponential Taylor Series
    e_to_x += x**i/math.factorial(i)
    
print(e_to_x)
"""
#Simple code, but has no feature error control

import math 
# Define Acceptable Error Level
E_s = 0.000005
x = 0.5
e_to_x = 1e-10 # Ensure that no divison by 0 can take place on first iteration

for i in range(10):
    e_to_x_old = e_to_x # Storing previous values
    e_to_x += x**i/math.factorial(i) #Exponential Taylor Series and New value
    Abs_e = e_to_x - e_to_x_old # Absolute Error using approximate values
    Rel_e = (Abs_e) / e_to_x_old # Relative Error using approximate values
    if Rel_e < E_s:
        break

print("E to the x in 5, 10 ,20 decimal places are as follows:", "%0.5f" % e_to_x, "%0.10f" % e_to_x,"%0.20f" % e_to_x)
print("Number of iterations required", i)
print("The approximate Absolute error is", Abs_e)
print("The Approximate Relative error is", Rel_e)
    
    
    