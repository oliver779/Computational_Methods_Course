# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 20:34:32 2021

@author: Oliver
"""
"""Multivariate Optimization"""
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
print(f'sys executable {sys.executable}')
"""Steepest Ascent"""
def HOpt(F,dFx,dFy,x,y):
    import sympy as sym
    from sympy import symbols, solve
    import matplotlib.pyplot as plt

    hsym = symbols('hsym')

    xlist = []
    ylist = []
    flist = []
    dfxlist = []
    dfylist = []

    for i in range(0, 10, 1):
        xold = x
        yold = y

        dfx = dFx(x)
        dfy = dFy(y)

        #Create a function for the path to the top of the mountain.
        g = F(x+dfx*hsym, y+dfy*hsym)
        hexpr = sym.diff(g, hsym)

        hsolved = solve(hexpr)
        hopt = hsolved[0]

        x = xold + hopt*dfx
        y = yold + hopt*dfy

        Fxy = F(x, y)
        dfx = dFx(x)
        dfy = dFy(y)

        xlist.append(x)
        ylist.append(y)
        flist.append(Fxy)
        dfxlist.append(dfx)
        dfylist.append(dfy)

        if dfx <= 0.0001 and dfy <= 0.0001:
            break

    print(x, y, Fxy, dfx, dfy)

def F(x,y):
    return 2*x*y+2*x-x**2-2*y**2

def dFx(x):
    return 2*y+2-2*x

def dFy(y):
    return 2*x-4*y

x = 1.
y = 1.
print(HOpt(F, dFx, dFy, x, y))

# ADD MICROSOFT EXCEL HERE

"""Constrained Optimization using Laplace Multiplier"""

cost_social = 25
cost_tv = 250
budget = 2500

#lets get the minimum and maximum number of campaigns:
social_min = 0
social_max = budget / cost_social

tv_min = 0
tv_max = budget / cost_tv

# if we fix the number of tv campaings, we know the number of social campaigns left to buy by inverting the formula
def n_social(n_tv, budget):
    return (budget - 250 * n_tv) / 25

# if we fix the number of social campaings, we know the number of tv campaigns left to buy by inverting the formula
def n_tv(n_social, budget):
    return (budget - 25 * n_social) / 250

social_x = np.linspace(social_min, social_max, 100)
tv_y = n_tv(social_x, budget)

plt.figure(figsize=(10,5))
plt.plot(social_x, tv_y)
plt.xlabel('Number of social campaigns')
plt.ylabel('Number of tv campaigns')
plt.title('Possible ways of spending the budget')
plt.show()

def revenues(social, tv):
    return social**(3/4) * tv**(1/4) * 7

from mpl_toolkits.mplot3d import Axes3D
social_axis = np.linspace(social_min, social_max, 100)
tv_axis = np.linspace(tv_min, tv_max, 100)
social_grid, tv_grid = np.meshgrid(social_axis, tv_axis)

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_surface(tv_grid, social_grid, revenues(social_grid, tv_grid))

ax.plot(tv_y, social_x, linewidth = 5, color = 'r')

ax.set_xlabel('Number of hours bought')
ax.set_ylabel('Number of materials bought')
ax.set_title('Possible ways of spending the budget')
plt.show()

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize = (15, 5))


social_axis = np.linspace(social_min, social_max, 100)
tv_axis = np.linspace(tv_max, tv_min, 100)
social_grid, tv_grid = np.meshgrid(social_axis, tv_axis)
    
im = ax_l.imshow(revenues(social_grid, tv_grid), aspect = 'auto', extent=[social_min, social_max, tv_min, tv_max])
ax_l.plot(social_axis, n_tv(social_axis, 2500), 'r')
ax_l.set_xlabel('Number of social campaigns bought')
ax_l.set_ylabel('Number of tv campaigns bought')
ax_l.set_title('Possible ways of spending the budget')


# The contours are showing how the intersection looks like

social_axis = np.linspace(social_min, social_max)
tv_axis = np.linspace(tv_min, tv_max)
social_grid, tv_grid = np.meshgrid(social_axis, tv_axis)

im2 = ax_r.contour(revenues(social_grid,tv_grid), extent=[social_min, social_max, tv_min, tv_max])
ax_r.plot(social_axis, n_tv(social_axis, 2500), 'r')
ax_r.set_xlabel('Number of social campaings bought')
ax_r.set_ylabel('Number of tv campaigns bought')
ax_r.set_title('Possible ways of spending the budget')

plt.colorbar(im,ax=ax_l)
plt.colorbar(im2,ax=ax_r)

plt.show()

from sympy import *

s, t, l = symbols('s t l')

solve([Eq((21/4)*((t**(1/4))/s**(1/4)) - 25*l, 0),
   Eq((7/4)*(s**(3/4)/t**(3/4)) - 250*l, 0),
   Eq(25*s+250*t - 2500, 0)], [s,t,l], simplify=False)

revenues(75, 2.5)

"""DOES NOT WORK PAST THIS LINE"""
# Lagrange multiplier from the lecture slides:
import numpy as np
from scipy.optimize import minimize
def objective(X):
    x, y = X
    return (x-5)**2 + (y-8)**2
#This is the constraint function that has lambda as a coefficient.
def eq(X):
    x, y = X
    return x*y - 5

def F(L):
    'Augmented Lagrange function'
    x, y, _lambda = L
    return objective([x, y]) + _lambda * eq([x, y])


from autograd import grad
import autograd.numpy as np

# Gradients of the Lagrange function
dfdL = grad(F, 0)

# Find L that returns all zeros in this function.
def obj(L):
    x, y, _lambda = L
    dFdx, dFdy, dFdlam = dfdL(L)
    return [dFdx, dFdy, eq([x, y])]
from scipy.optimize import fsolve
x, y, _lam = fsolve(obj, [0.0, 0.0, 0.0])
print(f'The answer is at {x, y}')


# solving an economic problem
def objective(X):
    x, y = X
    return (160*x**0.66)*(y**0.33)
#This is the constraint function that has lambda as a coefficient.
def eq(X):
    x, y = X
    return 20*x + 0.15*y - 20000.


def G(L):
    'Augmented Lagrange function'
    x, y, _lambda = L
    return objective([x, y]) + _lambda * eq([x, y])

# Gradients of the Lagrange function
dfdL = grad(G, 0)
# Find L that returns all zeros in this function.
def obj(L):
    x, y, _lambda = L
    dFdx, dFdy, dFdlam = dfdL(L)
    return [dFdx, dFdy, eq([x, y])]
from scipy.optimize import fsolve
x, y, _lam = fsolve(obj, [1., 1., 1.0])
print(f'The answer is at {x, y}')












