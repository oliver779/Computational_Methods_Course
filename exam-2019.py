# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 14:42:33 2021

@author: Oliver
"""
#%% IMPORTING VARIABLES

import matplotlib.pyplot as plt
import math
import numpy as np
import sympy as sp
import timeit

#%% QUESTION 1
sm = 1500
cm = 12
min_x = -100
max_x = 100

def function(x):
    return x**4 + 2*cm*x**3 +3*sm*x**2+cm*sm*x+sm**2
x = np.linspace(min_x,max_x,100000)
func_values = []
for i in x:
    func_values.append(function(i))
plt.plot(x,func_values)
plt.hlines(0,min_x,max_x)
plt.show()


#%% QUESTION 2

from sympy import *
 
x = Symbol('x')
p = Symbol('p')
z = integrate(sin(x)*cos(p*x),(x, 0, pi))
print(z)
equation = lambda x: -cos(pi*x)/(x**2 - 1)

print(equation(3.14159)-equation(0))

def f(x):
    return -np.cos(3.14159*x)/(x**2 - 1)
x = np.linspace(-1, 1)
fig, ax = plt.subplots()
ax.plot(x, f(x), label='target')
ax.grid()

def fprime(x):
    return (3.14159*(x**2-1)*np.sin(3.14159*x)+2*x+2*x*np.cos(3.14159*x))/(1-x**2)**2
def fsecond(x):
    return -(4*3.14159*x*np.sin(3.14159*x))/(1-x**2)**2-(3.14159**2*np.cos(3.14159*x))/(1-x**2)+((8*x**2)/(1-x**2)**3+2/(1-x**2)**2)*(1+np.cos(3.14159*x))

def quadratic_approx(x, x0, f, fprime, fsecond):
    return f(x0)+fprime(x0)*(x-x0)+0.5*fsecond(x0)*(x-x0)**2

x = np.linspace(-1, 1)
fig, ax = plt.subplots()
ax.plot(x, f(x), label='target')
ax.grid()
ax.plot(x, quadratic_approx(x, 0, f, fprime, fsecond), color='red', label='quadratic approximation')
ax.set_ylim([-5,3])

plt.legend()

def newton(x0, fprime, fsecond, maxiter=100, eps=0.0001):
    x=x0
    for i in range(maxiter):
        xnew=x-(fprime(x)/fsecond(x))
        if xnew-x<eps:
            return xnew
            print('converged')
            break
        x = xnew
    return x

x_star=newton(0, fprime, fsecond)
print(f'The value of p that minimizes this equation is: {x_star}')
fig, ax = plt.subplots()
ax.plot(x, f(x), label='target')
ax.grid()
ax.plot(x, quadratic_approx(x, x_star , f, fprime, fsecond), color='red', label='quadratic approximation')
ax.set_ylim([-5,3])

plt.legend()

#%% QUESTION 3
from scipy.interpolate import interp2d, interp1d
import numpy as np
import matplotlib.pyplot as plt

ref_theta = [-30,0,30]
ref_R = [6870,6728,6615]

r_f = interp1d(ref_theta,ref_R,'quadratic')
values_R  = []
theta_values =  np.linspace(-30,30,10000)
for i in theta_values:
    values_R.append(r_f(i))
    if i == 30:
        print(r_f(i))
        break
    
plt.plot(theta_values,values_R, label = 'interpolation')
plt.plot(ref_theta,ref_R, label = 'original')
plt.legend()
plt.show()
plt.plot(values_R,theta_values, label = 'interpolation')
plt.plot(ref_R,ref_theta, label = 'original')
plt.legend()
plt.show()

















