# This is the place where we write down all of the functions and variables for now
# possibly add functions regarding colors and functions for inputs maybe?

# Importing the necessary packages

from math import sqrt
from random import gauss
import matplotlib.pyplot as plt
import random
from scipy.interpolate import interp2d

# Variable meanings

# n - the number of particles to be on the grid
# mean - just mean which can be changed by the user
# variance - just variance which can be changed by the user
# xllim - x lower limit for the grid that can be changed by the user
# xulim - x upper limit for the grid that can be changed by the user
# yllim - y lower limit for the grid that can be changed by the user
# yulim - y upper limit for the grid that can be changed by the user
# x_database - values of x positions of the velocity file
# y_database - values of y positions of the velocity file
# u_database - values of x velocities of the velocity file
# v_database - values of y velocities of the velocity file
# D - diffusivity rating
# h - The timestep

# Variables needed

x = [] # x position of the random points
y = [] # y position of the random points
x_new = []
y_new = []
x_database = []
y_database = []
u_database = []
v_database = []
mean, variance, D, h = 0, 1, 0.01, 0.0005
xllim = -1
xulim = 1
yllim = -1
yulim = 1

# Random brownian motion part function

def epsilon_xy(mean, variance):
    return sqrt(2*D)*sqrt(h)*gauss(mean, sqrt(variance))

# Generator of Random Uniform points

def random_initial_values(n,xllim,xulim,yllim,yulim):
    for i in range(n):
        x.append(random.uniform(xllim + 0.04, xulim - 0.04))
        y.append(random.uniform(yllim + 0.04, yulim - 0.04))
    return

# The plotting function 

def plot(n,xllim,xulim,yllim,yulim):
    plt.figure()
    plt.grid( linestyle='--', linewidth=1)
    plt.scatter(x, y, s=10)
    plt.axis([xllim, xulim, yllim, yulim])
    plt.xlabel("x - axis")
    plt.ylabel("y - axis")
    return plt.show()

# opening the velocity file and getting seperated values from it

def velocity_profile_separation_into_arrays():
    i=0
    f = open("velocityCMM3.dat", "r")
    for line in f:          
       for number in line.split():
           if i == 0:              
               x_database.append(float(number))
               i += 1
           elif i == 1:
               y_database.append(float(number))
               i += 1
           elif i == 2:
               u_database.append(float(number))
               i += 1
           else:
               v_database.append(float(number))
               i = 0
    return 
velocity_profile_separation_into_arrays()

# interpolation

u_f = interp2d(x_database,y_database,u_database,'linear')
v_f = interp2d(x_database,y_database,v_database,'linear')

# velocity in the x direction function

def u(x,y):
    return u_f(x,y)

# velocity in the y direction function

def v(x,y):
    return v_f(x,y)

# Euler maruyama method for the x values of the points

def Euler_Maruyama_x_position(n,x_position,y_position):
    if xulim - 0.03 <= x_position  or  x_position <= xllim + 0.03:
        new_x_position = x_position - u(x_position,y_position)*h + epsilon_xy(mean, variance)
    else:
        new_x_position = x_position + u(x_position,y_position)*h + epsilon_xy(mean, variance)
    x_new.append(new_x_position)
    return 

# Euler maruyama method for the y values of the points 

def Euler_Maruyama_y_position(n,x_position,y_position):
    if yulim - 0.03 <= y_position  or  y_position <= yllim + 0.03:
        new_y_position = y_position - v(x_position,y_position)*h + epsilon_xy(mean, variance)
    else:
        new_y_position = y_position + v(x_position,y_position)*h + epsilon_xy(mean, variance) 
    y_new.append(new_y_position)
    return 

# clearing of arrays function (x,y)

def clear():
    x.clear()
    y.clear()
    return

# clearing of arrays function #2 (new_x,new_y)

def clear2():
    x_new.clear()
    y_new.clear()
    return

# The function that combines moving the new values from euler into previous arrays and clears them from everything so 
# that thecomputer can quickly calculate things without remembering useless information

def counter(n,x,y):
    for element, element1 in zip(x, y):
        Euler_Maruyama_x_position(n, element, element1)
        Euler_Maruyama_y_position(n, element, element1)
    clear()
    for i in range(0,len(x_new)):
        x.append(x_new[i])
        y.append(y_new[i]) 
    clear2()
    return x,y    



















