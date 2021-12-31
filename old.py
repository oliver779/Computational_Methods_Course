# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 01:13:19 2021

@author: Oliver
"""

# This is the place where we write down all of the functions

# Importing the necessary packages

import math
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

# Variables needed
x = [] # x position of the random points
y = [] # y position of the random points
x_new = [] # new x coordinate of the random point
y_new = [] # new y coordinate of the random point
x_database = []
y_database = []
u_database = []
v_database = []
mean, variance, D, h = 0, 1, 0.01, 0.0005
# interpolate the velocity field into the region

def epsilon_x(mean, variance):
    return gauss(mean, sqrt(variance))
def epsilon_y(mean, variance):
    return gauss(mean, sqrt(variance))

# Generator of Random Uniform points

def random_initial_values(n,xllim,xulim,yllim,yulim):
    for i in range(n):
        x.append(random.uniform(xllim + 0.02, xulim - 0.02))
        y.append(random.uniform(yllim + 0.02, yulim - 0.02))
    return

# The plotting function 

def plot(n,xllim,xulim,yllim,yulim):
    # draw the plot
    plt.figure()
    plt.grid( linestyle='--', linewidth=1)
    plt.scatter(x, y, s=10)
    plt.axis([xllim, xulim, yllim, yulim])
    #plt.axis('equal')
    plt.xlabel("x - axis")
    plt.ylabel("y - axis")
    return plt.show()
# opening the velocity profile and getting seperated values from it

def velocity_profile_separation_into_arrays():
    i=0
    f = open("velocityCMM3.dat", "r")
    for line in f:   
       # reading each number        
       for number in line.split():
           #print(number)
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
    return x_database, y_database, u_database, v_database

# velocity in the x direction function

def u(x,y):
    u_f = interp2d(x_database,y_database,u_database,'linear')
    return u_f(x,y)

# velocity in the y direction function

def v(x,y):
    v_f = interp2d(x_database,y_database,v_database,'linear')
    return v_f(x,y)

# Euler maruyama method for the x values of the points

def Euler_Maruyama_x_position(n,x_position,y_position):
    new_x_position = x_position + u(x_position,y_position)*h + math.sqrt(2*D)*sqrt(h)*epsilon_x(mean, variance)
    x[x.index(x_position)] = new_x_position
    return x

# Euler maruyama method for the y values of the points 

def Euler_Maruyama_y_position(n,x_position,y_position):
    new_y_position = y_position + v(x_position,y_position)*h + math.sqrt(2*D)*sqrt(h)*epsilon_x(mean, variance)
    y[y.index(y_position)] = new_y_position
    return y

# clearing of arrays function (x,y)

# def clear():
#     x.clear()
#     y.clear()
#     return

# # clearing of arrays function #2 (new_x,new_y)

# def clear2():
#     x_new.clear()
#     y_new.clear()
#     return

# The function that combines moving the new values from euler into previous arrays and clears them from everything so 
#that thecomputer can quickly calculate things without remembering useless information

def counter(n,x,y):
    for element, element1 in zip(x, y):
        Euler_Maruyama_x_position(n, element, element)
        Euler_Maruyama_y_position(n, element1, element1)
    # clear()
    # for i in range(0,len(x_new)):
    #     x.append(x_new[i])
    #     y.append(y_new[i]) 
    # clear2()
    return x,y
