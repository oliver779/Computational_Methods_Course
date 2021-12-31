# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 15:13:24 2021

@author: Oliver
"""
import math

def f(x):
    return x**3 + x**2 + x + 1
# xr = 1
# h = 0.1
# eps = 0.01
# maxit = 100
# def Muller(xr,h,eps,maxit):
#     iter = 0
#     x2 =xr
#     x1 = xr + h*xr
#     x0 = xr - h*xr
#     while True:
#         iter += 1
#         h0 = x1-x0
#         h1 = x2- x1
#         d0 = (f(x1)-f(x0))/h0
#         d1 = (f(x2)-f(x1))/h1
#         a = (d1-d0)/(h1+h0)
#         b = a*h1 + d1
#         c = f(x2)
#         rad = math.sqrt(b**2 - 4*a*c)
#         if abs(b+rad) > abs(b-rad):
#             den = b+rad
#         else:
#             den = b-rad
#         dxr = -2*c / den
#         xr = x2 +dxr
#         print(iter, xr)
#         if (abs(dxr) < eps*xr or maxit <= iter):
#             break
#         x0 = x1
#         x1 = x2
#         x2 = xr
# print(Muller(xr,h,eps,maxit)

def Muller(x_r, h, eps, maxit):
    iter = 0
    x2 = x_r
    x1 = x_r + h * x_r
    x0 = x_r - h * x_r
    while True:
        iter += 1
        h0 = x1 - x0
        h1 = x2 - x1
        d0 = (f(x1) - f(x0)) / h0
        d1 = (f(x2) - f(x1)) / h1
        a = (d1 - d0) / (h1 + h0)
        b = a * h1 + d1
        c = f(x2)
        rad = math.sqrt(abs(b**2 - 4*a*c))
        if abs(b + rad) > abs(b - rad):
            den = b + rad
        else:
            den = b - rad
        dx_r = -2 * c / den
        x_r = x2 + dx_r
        print(iter, x_r)
        if (abs(dx_r) < eps * x_r or iter >= maxit):
            break
        x0 = x1
        x1 = x2
        x2 = x_r
        return None
x_r = 1
h = 0.1
eps = 0.01
maxit = 100   
solution = Muller(x_r, h, eps, maxit)
print(solution)