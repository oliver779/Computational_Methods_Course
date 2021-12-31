# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 13:05:36 2021

@author: Oliver
"""
import math
import matplotlib.pyplot as plt
import numpy as np

def f(x):
    z = 2*math.sin(x)-x**2/10
    return z

# x = []
# y = []
# for i in range(0,3):
#     x.append(i)
#     y.append(f(i))
# plt.plot(x,y)

# maxit = 10
# es = 5
# x_l = -10
# x_u = 10
# f_x = 0

# def golden_ratio():
#     R = (5**(1/2) - 1)/2
#     d = R*(x_u-x_l)
#     x_1 = x_l + d
#     x_2 = x_u - d
#     f1 = f(x_1)
#     f2 = f(x_2)
#     counter = 1
#     if f1 > f2:
#         x_opt = x_1
#         f_x = f1
#         return x_opt
#     else:
#         x_opt = x_2
#         f_x = f2
#         return x_opt
        
        
        
# while True:
# d = R*d
# x_int = x_u-x_l
# if f1 > f2:
#     x_l = x_2
#     x_2 = x_1
#     x_1 = x_l + d
#     f2 = f1
#     f1 = f(x_1)
# else:
#     x_u = x_1
#     x_1 = x_2
#     x_2 = x_u - d
#     f1 = f2
#     f2 = f(x_2)
# counter +=1
# if f1<f2:
#     x_opt = x_1
#     f_x = f1
# else:
#     x_opt = x_2
#     f_x = f2
# counter+=1
# if f1>f2:
#     x_opt = x_2
#     f_x = f1
# else:
#     x_opt = x_2
#     f_x = f2
# if x_opt !=0:
#     ea = (1-R)*abs(x_int/x_opt)*100
# if ea <=es or counter>=maxit:
#     break
# return x_opt

        
        
# print(golden_ratio(x_l,x_u,maxit,es))





guess_x_lower = 0
guess_x_upper = 4
step = (math.sqrt(5) - 1)/2*(guess_x_upper-guess_x_lower)
x_1 = guess_x_lower + step
x_2 = guess_x_upper - step
counter = 0
error = 1-1
while True:
    if f(x_1)>f(x_2):
        x_2 = x_1
    if f(x_1)<f(x_2):
        guess_x_lower = guess_x_upper
    x_1 = guess_x_lower + step
    x_1 = guess_x_upper
    x_2 = x_2 - step
    counter +=1
    if counter == 8:
        print(x_2)
        print(f(x_2))
        print(x_1)
        print(f(x_1))
        break





































