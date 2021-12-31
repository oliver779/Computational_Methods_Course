# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 15:37:21 2021

@author: hlack
"""

import numpy as np
import matplotlib.pyplot as plt
import random
n = 1000
x = []
y = []
x_c = []
y_c = []

def random_initial_values(n):
    for i in range(n):
        x.append(random.uniform(-1, 1))
        y.append(random.uniform(-1, 1))
    for i in range(n):
        if (x[i])**2 + (y[i])**2 <= 0.5**2:
            x_c.append(x[i])
            y_c.append(y[i])


random_initial_values(n)

xlim = [-1, 1]
ylim = [-1,1]


a,xe, ye = np.histogram2d(x,y, bins= 16, range = (xlim,ylim) )
b,xe1,ye1  = np.histogram2d(x_c, y_c, bins = 16, range = (xlim,ylim))
a = np.where(a == 0, 1, a)
concentration = b/a

dx, dy = 0.25, 0.25
Y, X = np.mgrid[slice(-1, 1 + dy, dy),
                slice(-1, 1 + dx, dx)]


plt.figure()
plt.grid( linestyle='--', linewidth=1)
c = plt.pcolormesh(X, Y, concentration, cmap = 'rainbow')
plt.colorbar(c)


plt.show()
