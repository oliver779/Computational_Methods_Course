

import matplotlib.pyplot as plt
import math as m
import sympy as sp

x = sp.Symbol('x')

"""Now that we just need to make the differentiation to work"""

# x - distance along beam (m)
# y - deflection (m)
# theta(x) - slope(m/m)
# E - modulus of elasticity (Pa = N/m**2)
# I - Moment of inertia (m**4)
# M(x) - moment (Nm)
# V(x) - shear (N)
# w(x) - distributed load (N/m)

E = 200
I = 0.0003
w_0 = 2.5
y_0 = 0
y_L = 0
x = 1
dx = 0.125
L = 3

def theta_x(L):
    return w_0/(120*E*I*L)*(-5*x**4+6*L**2*x**2 - L**4)

def M_x():
    return w_0/(120*L)*(-20*x*3+6*L**2*2*x)

def V_x():
    return w_0/(120*L)*(-60*x*2+6*L**2*2)

def w_x():
    return -(w_0/(120*L)*(-120*x))

V_x_values = []
theta_values = []
M_x_values = []
w_x_values = []

while L>=0:
    if L == 0:
        break
    w_x_values.append(w_x())
    M_x_values.append(M_x())
    theta_values.append(theta_x(L))
    V_x_values.append(V_x())
    L +=-dx
    print(L)
    
print("Theta values")
print(theta_values)
print("Moment values")
print(M_x_values)
print("Shear Values")
print(V_x_values)
print("Distributed Load")
print(w_x_values)
# dy/dx = theta(x)    
# dtheta/dx = M(x)/(EI)
# dM/dx = v(x)
# dV/dx = -w(x)

# plt.plot(x,slope)
# plt.show()