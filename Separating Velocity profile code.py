# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 14:30:51 2021

@author: hlack
"""

import numpy as np

f = open("velocityCMM3.dat", "r")
'''
count = 0
# Strips the newline character
for line in Lines:
    count += 1
    print("Line{}: {}".format(count, line.strip()))
    '''
i = 0
x = []
y = []
u = []
v = []
#print("xxxxx")
 # reading each line    
for line in f:   
   # reading each number        
   for number in line.split():
       #print(number)
       if i == 0:
           x.append(number)
           i += 1
       elif i == 1:
           y.append(number)
           i += 1
       elif i == 2:
           u.append(number)
           i += 1
       else:
           v.append(number)
           i = 0
print(v[11])
print(y[26])
print(u[5])
print(x[15])

def velocity_profile_separation_into_arrays():
    i=0
    x_database = []
    y_database = []
    u_database = []
    v_database = []
    f = open("velocityCMM3.dat", "r")
    for line in f:   
       # reading each number        
       for number in line.split():
           #print(number)
           if i == 0:
               x_database.append(number)
               i += 1
           elif i == 1:
               y_database.append(number)
               i += 1
           elif i == 2:
               u_database.append(number)
               i += 1
           else:
               v_database.append(number)
               i = 0
    return x_database, y_database, u_database, v_database