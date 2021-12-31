# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 11:05:25 2021

@author: Oscar
"""

"Tutorial 10 "
import math as m
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

#%% Question 1: Deflection of a beam
# A) use numerical integration to compute the deflection

# Define the variables and equations

# Length of beam
L = 3 #m
# interval size
delta_x = 0.125 #m
interval = L/delta_x
# Modulus of Elasticity
E = 2e11 # 200 Gpa ~ 2e5 MPa
# Moment of inertia
I = 3e-4 # m^4
# Distributed load
w0 = 250000 # 2.5N/cm or 250000N/m

# Define Functions using Sympy
x = sp.Symbol('x')
f = (w0*(-5 * x**4 + (6 * L**2) * x**2 -L**4))/(120*E*I*L)
theta = sp.lambdify(x, f)

# Function to perform calculations 
def SimpsonsThird(lower_limit, upper_limit, interval): 
     
    interval_size = (float(upper_limit - lower_limit) / interval) #dx or h`
    sum = theta(lower_limit) + theta(upper_limit); 
       
    # Calculates value till integral limit 
    for i in range(1, interval ):
        
        if (i % 3 == 0): 
            sum = sum + 2 * theta(lower_limit + i * interval_size) 
        else: 
            sum = sum + 3 * theta(lower_limit + i * interval_size) 
      
    return ((float( 3 * interval_size) / 8 ) * sum ) 
  
integral_res = SimpsonsThird(0, L/2, 24) 
print('The value of the deflection at the midpoint of the beam (x = 1.5 m) is ', integral_res)

#%% Question 1B): Numerical differention to computed moment and shear

# Moment = d/dx (theta)*EI
# Shear =d/dx (Moment)

g = sp.diff(f, x)
Moment = sp.lambdify(x, g)
solution = Moment(3)*(E*I)
print('The Moment is %.2f' %solution, 'at x = 3 m.')

v = sp.diff(f, x, 2)
Shear = sp.lambdify(x, v)
force = Shear(3)*E*I
print('The Shear is %.2f' %force, 'at x = 3 m ')

