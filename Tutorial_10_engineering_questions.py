# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 20:49:22 2021

@author: Oliver

"""
import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt
w_0=250
E=200000
I=300000000
delta_x =125 # mm
L = 3000 #mm

y = np.linspace(0,3000,24000)
def theta_x(x):
    return w_0/(120*E*I*L) *(-5*x**4 +6*L**2*x**2 - L**4)

dydx = theta_x(0)

"E*I*dthetadx = M(x)"
"dMdx = V(x)"
"dVdx = -w(x)"

def calculate_dx (a, b, n):
    return (b-a)/float(n)

def rect_rule (f, a, b, n):
    total = 0.0
    dx = calculate_dx(a, b, n)
    for k in range (0, n):
        total = total + f((a + (k*dx)))
    return dx*total
print(rect_rule(theta_x, 0, 1.5, 24))

deflection_values = []

for i in y:
    deflection_values.append(rect_rule(theta_x,0,i,24))
plt.title("THIS IS THE GRAPH")   
plt.plot(y,deflection_values)
plt.show()

# DERIVATIVES NOW
z = sp.Symbol('z')
x = sp.Symbol('x')
f = sp.Function('f')(x)
f = w_0/(120*E*I*L) *(-5*x**4 +6*L**2*x**2 - L**4)
M_x = sp.diff(f)/(E*I)
v = M_x.evalf(subs = {x:1.5})
              
print(f'The Moment at midpoint is: {v}')
V_x = sp.diff(M_x)
w_x = sp.diff(V_x)
for i in y:
    if i==3:
        print(0)
        break
    print(w_x.evalf(subs = {x:i}))
    
print(f'Moment is given by the following equation: {M_x}')
print(f'Shear is given by the following equation: {V_x}')
print(f'Load is given by the following equation: {V_x}')


# NEXT QUESTION 
# SPHERICAL TANK HAS A CIRCULAR ORIFICE IN ITS BOTTOM THROUGH
# WHICH THE LIQUID FLOWS OUT
# inputs

# functions that returns dy/dx
# i.e. the equation we want to solve: dy/dx = - y
def model(y,x):
    C=0.55
    r_ori = 0.3
    A=(r_ori/2)**2*math.pi
    g=9.81
    r=1.5
    dydx = -(C*A*(2*g*y)**0.5)/(2*math.pi*r*y-math.pi*y**2)
    return dydx

# initial conditions
x0 = 0
y0 = 2.75
# total solution interval
x_final = 100
# step size
h = 0.1
# ------------------------------------------------------

# ------------------------------------------------------
# Euler method

# number of steps
n_step = math.ceil(x_final/h)

# Definition of arrays to store the solution
y_eul = np.zeros(n_step+1)
x_eul = np.zeros(n_step+1)

# Initialize first element of solution arrays 
# with initial condition
y_eul[0] = y0
x_eul[0] = x0 

# Populate the x array
for i in range(n_step):
    x_eul[i+1]  = x_eul[i]  + h

# Apply Euler method n_step times
for i in range(n_step):
    # compute the slope using the differential equation
    slope = model(y_eul[i],x_eul[i]) 
    # use the Euler method
    y_eul[i+1] = y_eul[i] + h * slope  
    if y_eul[i+1] < 0:
        break
    # print( x_eul[i+1], y_eul[i+1])
# ------------------------------------------------------

# ------------------------------------------------------
# plot results
plt.plot(x_eul, y_eul , 'b.-',)
plt.xlabel('time (s)')
plt.ylabel('y(x)')
plt.show()
# ------------------------------------------------------

# FINAL QUESTION OF THE TUTORIAL

ff,zz,EE,II,LL,zz = sp.symbols('ff,zz,EE,II,LL,zz')


z = 30 #HEIGHT AT THE END WHICH IS EQUAL TO L
f=60
E=1.25*10**8
I=0.05
L=30
    
g = ff*(LL-zz)**2/(2*EE*II)


Integral_1 = sp.integrate(g,zz)
Integral_2 = sp.integrate(Integral_1,zz)
print(Integral_1)
print(Integral_1.subs([(zz,z),(ff,f),(EE,E),(II,I),(LL,L)]))
print(Integral_2)
print(Integral_2.subs([(zz,z),(ff,f),(EE,E),(II,I),(LL,L)]))




