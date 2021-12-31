# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 22:05:32 2021

@author: Oliver
"""
import sympy  as sp
import numpy as np
import matplotlib.pyplot as plt
import math

E = 200000 #MPA
I = 300000000 #mm^4
w0 = 250 #N/mm
dx = 125 #mm
y0 = 0
yL = 0
L = 3000
y = np.linspace(0,3000,24000)
x = sp.Symbol('x')
f = sp.Function('f')(x)
f = w0/(120*E*I*L)*(-5*x**4+6*L**2*x**2-L**4)

df = sp.diff(f)
M_x = E*I*df
dMdxV_x = sp.diff(E*I*df)
dVdxw_x = -sp.diff(dMdxV_x)


"E*I*dthetadx = M(x)"
"dMdx = V(x)"
"dVdx = -w(x)"
deflection = sp.integrate(f)
print(deflection)
# deflection_values = []
# for i in y:
#     v = deflection.evalf(subs = {x:i})
#     deflection_values.append(v)
    
# plt.plot(y,deflection_values)
# plt.xlabel('Length of bar (mm)')
# plt.ylabel('Deflection (mm)')
# plt.show()   

print(f'Moment is given by the following equation: {M_x}')
print(f'Shear is given by the following equation: {dMdxV_x}')
print(f'Load is given by the following equation: {dVdxw_x}')

#%% QUESTION 2

t_init = 0
x_final = 100

def model(y,x):
    g = 9.81
    C = 0.55
    h = 2.75
    r = 1.5
    D = 0.03
    A = 3.14*(D/2)**2
    Q = C*A*math.sqrt(2*g*y)
    dQdt = (-Q)/(2*3.14*y*r - 3.14*y**2)
    return dQdt


# ------------------------------------------------------
x0 = t_init
y0 = 2.75
# ------------------------------------------------------
# Euler method
x_final = 10000
# step size
x = 1
# number of steps
n_step = math.ceil(x_final/x)

# Definition of arrays to store the solution
y_eul = np.zeros(n_step+1)
x_eul = np.zeros(n_step+1)

# Initialize first element of solution arrays 
# with initial condition
y_eul[0] = 2.75
x_eul[0] = 0 

# Populate the x array
for i in range(n_step):
    x_eul[i+1]  = x_eul[i] + x

# Apply Euler method n_step times
for i in range(n_step):
    # compute the slope using the differential equation
    slope = model(y_eul[i],x_eul[i]) 
    # use the Euler method
    y_eul[i+1] = y_eul[i] + x * slope   
    if y_eul[i+1] <0:
        break

plt.plot(x_eul, y_eul , 'b.-',)
plt.xlabel('time (s)')
plt.ylabel('y(x)')
plt.show()

#%% QUESTION 3
print()
print("________________________________________________")
print()
f = 60
L = 30
E = 1.25*10**8
I = 0.05

y = sp.Function('y')
z = sp.Symbol('z')

y = f/(2*E*I)*(L-z)**2
int_y = sp.integrate(y,z)
d_int_y = sp.integrate(int_y)
print(f'The first integral is : {int_y}')
print(f'The second integral is : {d_int_y}')
d_of_mast = d_int_y.evalf(subs = {z:30})
print(f'The deflection of the mast at the peak is : {d_of_mast}m')
