"""Solving  a System of ODEs"""

"""Euler's Method""" """Approximation of functions/differential equations""" """NOT ROOT FINDING"""
# importing modules
import numpy as np
import matplotlib.pyplot as plt
import math
# ------------------------------------------------------
# functions that returns dy/dx
# i.e. the equation we want to solve: dy_j/dx = f_j(x,y_j)
# (j=[1,2] in this case)
def model(x,y_1,y_2):
    f_1 = -0.5 * y_1
    f_2 = 4.0 - 0.3 * y_2 - 0.1 * y_1
    return [f_1 , f_2]
# ------------------------------------------------------
# ------------------------------------------------------
# initial conditions
x0 = 0
y0_1 = 4
y0_2 = 6
# total solution interval
x_final = 2
# step size
h = 0.00001
# ------------------------------------------------------
# ------------------------------------------------------
# Euler method
# number of steps
n_step = math.ceil(x_final/h)
# Definition of arrays to store the solution
y_1_eul = np.zeros(n_step+1)
y_2_eul = np.zeros(n_step+1)
x_eul = np.zeros(n_step+1)
# Initialize first element of solution arrays
# with initial condition
y_1_eul[0] = y0_1
y_2_eul[0] = y0_2
x_eul[0] = x0
# Populate the x array
for i in range(n_step):
    x_eul[i+1] = x_eul[i] + h
# Apply Euler method n_step times
for i in range(n_step):
    # compute the slope using the differential equation
    [slope_1 , slope_2] = model(x_eul[i],y_1_eul[i],y_2_eul[i])
    # use the Euler method
    y_1_eul[i+1] = y_1_eul[i] + h * slope_1
    y_2_eul[i+1] = y_2_eul[i] + h * slope_2
print(f'Values of y_1: {y_1_eul[i]} and y_2: {y_2_eul[i]}')

"""Solve IVP METHOD""" """INTEGRATION"""
# importing modules
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import solve_ivp
# ------------------------------------------------------
# functions that returns dy/dx
# i.e. the equation we want to solve:
# dy_j/dx = f_j(x,y_j) (j=[1,2] in this case)
def model(x,y):
    y_1 = y[0]
    y_2 = y[1]
    f_1 = -0.5 * y_1
    f_2 = 4.0 - 0.3 * y_2 - 0.1 * y_1
    return [f_1 , f_2]
# ------------------------------------------------------
# ------------------------------------------------------
# initial conditions
x0 = 0
y0_1 = 4
y0_2 = 6
# total solution interval
x_final = 2
# step size
# not needed here. The solver solve_ivp
# will take care of finding the appropriate step
# -------------------------------------------------
# ------------------------------------------------------
# Apply solve_ivp method
y = solve_ivp(model, [0 , x_final] ,[y0_1 , y0_2])
# ------------------------------------------------------
# ------------------------------------------------------
# plot results
plt.plot(y.t,y.y[0,:] , 'b.-',y.t,y.y[1,:] , 'r-')
plt.xlabel('x')
plt.ylabel('y_1(x), y_2(x)')
plt.show()
# ------------------------------------------------------
# ------------------------------------------------------
# print results in a text file (for later use if needed)
file_name= 'output.dat'
f_io = open(file_name,'w')
n_step = len(y.t)
for i in range(n_step):
    s1 = str(i)
    s2 = str(y.t[i])
    s3 = str(y.y[0,i])
    s4 = str(y.y[1,i])
    s_tot = s1 + ' ' + s2 + ' ' + s3 + ' ' + s4
    f_io.write(s_tot + '\n')
f_io.close()
# ------------------------------------------------------

"""Implicit Euler Method""" """MAGIC BULLET FOR STIFF SYSTEMS"""

# importing modules
import numpy as np
import matplotlib.pyplot as plt
import math
# ------------------------------------------------------
# inputs
# functions that returns dy/dx
# i.e. the equation we want to solve: dy/dx = - y
def model2(y,x):
    dydx = -1000.0*y + 3000.0 - 2000.0*math.exp(-x)
    return dydx
# initial conditions
x0 = 0
y0 = 0
# total solution interval
x_final = 0.3
# step size
h = 0.05
# ------------------------------------------------------
# Secant method (a very compact version)
def secant_2(f, a, b, iterations):
    for i in range(iterations):
        c = a - f(a)*(b - a)/(f(b) - f(a))
        if abs(f(c)) < 1e-13:
            return c
        a = b
        b = c
    return c
# ------------------------------------------------------
# Euler implicit method
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
    x_eul[i+1] = x_eul[i] + h
# Apply implicit Euler method n_step times
for i in range(n_step):
    F = lambda y_i_plus_1: y_eul[i] + model2(y_i_plus_1,x_eul[i+1])*h - y_i_plus_1
    y_eul[i+1] = secant_2(F, y_eul[i],1.1*y_eul[i]+10**-3,10)
    print(y_eul[i+1])































