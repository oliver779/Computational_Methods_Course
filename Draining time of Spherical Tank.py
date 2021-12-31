# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 12:30:17 2021

@author: Oscar
"""

" Tutorial 10 Question 2: Spherical Tank"
import sympy as sp
import numpy as np
import math as m
import matplotlib.pyplot as plt

#%% Question 1: Flow rate through the orifice

#Flow rate Q = CAsqrt(2gH) (m^3/s)
    # Q is a rate of change, thus Q = dV/dt volume over time
    # Vol. sphere is 4/3 pi*r^3, Area of a circle is pi*r^2
    # Radius of the water level wrt to time can be found using pythagorean
    
# Define Parameters
# Grav. Accel
g = 9.81 #m/s^2
# Coefficient
C = 0.55
# Orifice Diameter
ori_d = 0.03 # 3cm
# Area of the orifice
A = (ori_d/2)**2 * m.pi
# Tank Diameter
tank_d = 3 #m
R = tank_d/2


def model(H, t): 
    Volume = -(C*A*m.sqrt(2*g*H))
    Area = (2*R*H*m.pi - H**2 * m.pi)
    Height = Volume/Area
    return Height

# Employing Euler's method 

# Initial Conditions
t0 = 0
h0 = 2.75

# total solution interval
x_final = 10000
# step size
h = 0.1

# number of steps
n_step = m.ceil(x_final/h)

# Definition of arrays to store solution
y_eul = np.zeros(n_step+1)
x_eul = np.zeros(n_step+1)

# Initialize first element of arrays
# with initial condition
y_eul[0] = h0
x_eul[0] = t0

# Populate the x array
for i in range(n_step):
    x_eul[i+1] = x_eul[i] + h
# Apply Euler method n_step times
for i in range(n_step):
    # compute the slope
    slope = model(y_eul[i],x_eul[i])
    # use the Euler method
    y_eul[i+1] = y_eul[i] + h * slope
    if y_eul[i+1] < 0:
        break
    print( x_eul[i+1], y_eul[i+1])

# plot results
plt.plot(x_eul, y_eul , 'b.-',)
plt.xlabel('time')
plt.ylabel('H(t)')
plt.show()
