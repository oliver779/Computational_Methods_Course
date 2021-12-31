# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 13:32:24 2021

@author: Oscar
"""

" Tutorial 10: Modeling the deflection of sail boat mast"
#%% Question 1
# if y =0 and dydz =0 at L=0, then z = L

import sympy as sp
import numpy as np
import math as m

# Define parameters

# Wind Force
f = 60
# Mast length
L = 30
# Modulus of Elasticity
E = 1.25 * (10**8)
# Moment of inertia 
I = 0.05

z = sp.Symbol('z')
dydz2 = f*(L-z)**2 / (2*E*I)
print(dydz2)

dydz = sp.integrate(dydz2, z)
print(dydz)

yz = sp.integrate(dydz, z)
print(yz)

deflection = sp.lambdify(z, yz)
print(deflection(30))