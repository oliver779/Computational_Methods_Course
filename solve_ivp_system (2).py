# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 15:32:20 2021

@author: Oscar
"""
"Tutorial 5 Numerical Simulation of chaos, the Lorenz System"
from scipy.integrate import solve_ivp
import numpy as np
import math
import matplotlib.pyplot as plt

# Define Function models
def model(t, y):
    rho = 28
    beta = 8/3
    sigma = 10
    
    y_1 = y[0]
    y_2 = y[1]
    y_3 = y[2]
    
    f1 = sigma * (y_2 - y_1)
    f2 = rho * y_1 - y_2 - y_1 * y_3
    f3 = -beta * y_3 + y_1 * y_2
    
    return [f1, f2, f3]

# Define Function models
def model2(t, y):
    rho = 10
    beta = 8/3
    sigma = 10
    
    y_1 = y[0]
    y_2 = y[1]
    y_3 = y[2]
    
    g1 = sigma * (y_2 - y_1)
    g2 = rho * y_1 - y_2 - y_1 * y_3
    g3 = -beta * y_3 + y_1 * y_2
    
    return [g1, g2, g3]

# Define initial Conditions
t = 0
t_f = 30

y0_1 = 5
y0_2 = 5
y0_3 = 5

# Define time interval
t_eval = np.linspace(t, t_f, num= 5000)

# Solution
y = solve_ivp(model, [t, t_f], [y0_1, y0_2, y0_3], t_eval= t_eval)
z = solve_ivp(model2, [t, t_f], [y0_1, y0_2, y0_3], t_eval= t_eval)
# ------------------------------------------------------
# Plot Results
plt.figure(1)
plt.plot(y.t, y.y[0,:] , 'b-', y.t,y.y[1,:],
         'r-', y.t, y.y[2,:], 'g-')
plt.xlabel('t')
plt.ylabel('y_1(t), y_2(t), y_3(t)')
# ------------------------------------------------------

# plot results
plt.figure(2)
plt.plot(y.y[0,:] ,y.y[1,:],'-')
plt.xlabel('y_1')
plt.ylabel('y_2')
# ------------------------------------------------------

# plot results
plt.figure(3)
plt.plot(y.y[0,:] ,y.y[2,:],'-')
plt.xlabel('y_1')
plt.ylabel('y_3')
# ------------------------------------------------------

plt.show()
# ------------------------------------------------------
# Plot Results
plt.figure(4)
plt.plot(z.t, z.y[0,:] , 'b-', z.t,z.y[1,:],
         'r-', z.t, z.y[2,:], 'g-')
plt.xlabel('t')
plt.ylabel('y_1(t), y_2(t), y_3(t)')
# ------------------------------------------------------

# plot results
plt.figure(5)
plt.plot(z.y[0,:] ,z.y[1,:],'-')
plt.xlabel('y_1')
plt.ylabel('y_2')
# ------------------------------------------------------

# plot results
plt.figure(6)
plt.plot(z.y[0,:] ,z.y[2,:],'-')
plt.xlabel('y_1')
plt.ylabel('y_3')

# print results in a text file (for later use if needed)
file_name= 'output.dat' 
f_io = open(file_name,'w') 
n_step = len(y.t)
for i in range(n_step):
    s1 = str(i)
    s2 = str(y.t[i])
    s3 = str(y.y[0,i])
    s4 = str(y.y[1,i])
    s_tot = s1 + ' ' + s2 + ' ' + s3  + ' ' + s4
    f_io.write(s_tot + '\n')
f_io.close()
# ------------------------------------------------------