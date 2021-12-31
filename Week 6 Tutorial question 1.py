# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 13:06:11 2021

@author: Oliver
"""
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

"""__________________REFERENCE SOLUTION________________"""
"""DONE"""

def integrand(x):
    return x**2+4*x-12
ans, err = quad(integrand, -10,10)
def integral(x):
    return 1/3*x**3+2*x**2-12*x
x= np.linspace(1,1000,8000)
plt.plot(x,integral(x),'g',label ='reference')
plt.legend()
plt.show()
print(f'The analytical solution to the integral is: {ans}')
print()


# Excercise 1: Implement the Midpoint/rectangular Rule code and execute
# it to integrate the function x^2+4x-12 in the domain -10<x<10


"""______________________________________________________________________"""
def calculate_dx(a,b,n):
    return (b-a)/float(n)
"""DONE"""

def rect_rule(f,a,b,n):
    total = 0.0
    dx = calculate_dx(a,b,n)
    t = []
    for i in range(0,n):
        total += abs(f((a+(i*dx))))
        t.append(total)
    plt.plot(range(n),t)
    plt.show()
    return dx*total

def f(x):
    return x**2+4*x-12

print(f' Excercise 1, midpoint/rectangular rule result: {rect_rule(f,-10,10,10000)}')

print(f'Analitycal - midpoint/rectangular rule result: {abs(rect_rule(f,-10,10,10000) - ans)}')

print(f'Relative error midpoint/rectangular result: {abs(ans - rect_rule(f,-10,10,10000) )/abs(rect_rule(f,-10,10,10000))}')
print()


"""______________________________________________________________________"""

#Excercise 2: Implement this trapezoid Rule code and execute it to integrate
# the function x**2 +4*x-12 in the domain -10<x<10 (enter inputs first)
"""NOT DONE PLOTTING"""
def trapz(f,a,b,N=50):
    x = np.linspace(a,b,N) # N+1 points make N subintervals
    print(x)
    y = f(x)
    y_right = y[1:] # right endpoints
    y_left = y[:-1] # left endpoints
    total = []
    for i in range(N+1,1,-1):
        dx = (b-a)/i
        Z = (dx/2) *np.sum(y_right+y_left)
        total.append(Z)
    plt.plot(x,total,label = 'traps')
    plt.legend()
    plt.show()
    # while i <= N:
    #     k += 2
    #     g.append((dx/2)*np.sum(f(k)+f(k+1))          
    # plt.plot(range(N),g)
    # plt.show
    return Z

a= -10
b= 10
n= 10000
print(f' The trapezoidal rule result is: {trapz(f,a,b,n)}')

print(f'Analitycal - trapezoidal rule: {abs(trapz(f,a,b,n) - ans)}')

print(f'Relative error trapezoidal rule result: {abs(ans - trapz(f,a,b,n) )/abs(trapz(f,a,b,n))}')
print()

"""______________________________________________________________________"""


# Excercise 3: Implement this Simpson's One Third Rule code and execute it
# to integrate the funciton x**2+4x-12 in the domain -10<x<10
"""NOT DONE PLOTTING"""
def simps(f,a,b,N=50):
    if N % 2 == 1:
        raise ValueError("N must be an even integer")
    t=[]
    x = np.linspace(a,b,N+1)
    y = f(x)
    for i in range(1,N+2,1):
        dx = (b-a)/i
        S = dx/3 * np.sum(y[0:-1:2] + 4*y[1::2] + y[2::2]) # s[i:j:k] - "slice of s from i to j with step k
        t.append(S)
    plt.plot(x,t)
    plt.show()
    return S # y[2::2] - start at the 2nd element and skip through in steps of 2 each time

f = lambda x: x**2+4*x-12
solution = simps(f,-10,10,10000)

print(f' The simpson one third rule solution: {solution}')

print(f'Analitycal - simpson one third rule: {abs(solution - ans)}')

print(f'Relative error simpson one third rule result: {abs(ans - solution )/abs(solution)}')
print()

"""______________________________________________________________________"""


# Exercise 4: Implement this Simpson’s three eightths Rule code and execute it to 
# integrate the function x2 +4x – 12 in the domain -10 < x < 10 (enter inputs first).
"""DONE"""
def func(x): 
    return abs(x**2+4*x-12)

def calculate(lower_limit, upper_limit, interval_limit ): 
    interval_size = (float(upper_limit - lower_limit) / interval_limit) 
    sum = func(lower_limit) + func(upper_limit); 
    # Calculates value till integral limit 
    n=0
    k = []
    t = []
    for i in range(1, interval_limit ): 
        if (i % 3 == 0): 
            k.append(n)
            n +=1
            sum = sum + 2 * func(lower_limit + i * interval_size)
            t.append(sum)
        else: 
            sum = sum + 3 * func(lower_limit + i * interval_size) 
    plt.plot(k,t,'r')
    return ((float( 3 * interval_size) / 8 ) * sum ) 

# driver function 

interval_limit = 10000
lower_limit = -10
upper_limit = 10
integral_res = calculate(lower_limit, upper_limit, interval_limit) 

# rounding the final answer to 6 decimal places 

print (f' The simpson three eigthths rule solution: {round(integral_res, 6)}') 

print(f'Analitycal - three eightths rule: {abs(round(integral_res, 6) - ans)}')

print(f'Relative error simpson three eightths rule result: {abs(ans - round(integral_res, 6) )/abs(round(integral_res, 6))}')

# Plot the evolution of the integral value as a function of the number of 
# integration intervals for each technique (Hint: you will have to modify each 
# code to run for different values of N and plot the integral obtained for each 
# run). Use arrays to store values and matplotlib to plot Integral v. N).














    