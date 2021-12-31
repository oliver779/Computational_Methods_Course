"""Euler Method for the equation dy/dx = -y"""

import numpy as np
import matplotlib.pyplot as plt
import math as m

x_initial = 0
y_initial = 1
x_final = 1
h = 0.02
step = m.ceil(x_final/h)


y_euler = []
x_euler = []

y_euler.append(y_initial)
x_euler.append(x_initial)

def equation(y,x):
    k = -1
    dydx = k*y
    return dydx

for i in range(step):
    element= x_euler[i] + h
    x_euler.append(element)
    slope = equation(y_euler[i],x_euler[i])
    element_2 = y_euler[i] + h*slope
    y_euler.append(element_2)
    

plt.plot(x_euler,y_euler)
plt.show()


































