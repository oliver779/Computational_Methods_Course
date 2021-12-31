# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 09:11:12 2020

@author: emc1977
"""

import numpy as np
from scipy import optimize

def f(x):   # The rosenbrock function
    return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2

def jacobian(x):
    return np.array((-2*.5*(1 - x[0]) - 4*x[0]*(x[1] - x[0]**2), 2*(x[1] - x[0]**2)))

solution = optimize.minimize(f, [2,-1], method="Newton-CG", jac=jacobian) 
print(solution)
