# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 14:29:28 2021

@author: Oliver
"""

import sympy as sp
import math
import numpy as np
import matplotlib.pyplot as plt
# sympy does not like numpy

y = sp.Symbol('y')
x = sp.Symbol('x')
f = sp.Function('f')(x)
g = sp.Function('g')(x,y)
f =  x**2 +x+5
df = sp.diff(f)
ddf = sp.diff(f,x,2)
dddf = sp.diff(f,x,3)
print(f.evalf(subs = {x:10}))
print(dddf.evalf(subs = {x:10}))
g = x*y + y**2 +x**3 
z = np.linspace(-10,10,100)

def random():
    return np.random.random_integers(1,20)

values = []
for i in z:
    d = g.evalf(subs = {x:i, y:random()})
    print(d)
    values.append(d)
    
plt.plot(z,values)
plt.show()











