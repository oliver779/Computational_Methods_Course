# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 15:31:52 2021

@author: Oliver
"""
import matplotlib.pyplot as plt
import timeit
x=1.5
y=3.5
uxy = lambda x,y: x**2 + y*x - 10
vxy = lambda x,y: y + 3*x*y**2 - 57
epsilon=0.00000000000001
max_iter=1000
values_x = []
values_y = []
def dvdx(y):
    return 3*y**2
def dvdy(x,y):
    return 1 + 6*x*y
def dudy(x):
    return x
def dudx(x,y):
    return 2*x + y
def determinant(x,y):
    return dudx(x,y)*dvdy(x,y) - dudy(x)*dvdx(y)
def f(x,y):
    a = dvdy(x,y)
    b = dudy(x)
    return  lambda x,y: (x**2 + y*x - 10)*a - (y + 3*x*y**2 - 57)*b
def g(x,y):
    c = dudx(x,y)
    d = dvdx(y)
    return lambda x,y: (y + 3*x*y**2 - 57)*c - (x**2 + y*x - 10)*d

def newton(f,g,df,x,y,epsilon,max_iter):
    for n in range(0,max_iter):
        print(f"The values of x,y respectively in the {n}th for loop are: {x,y}")
        values_x.append(x)
        values_y.append(y)
        if abs(float(f(x,y))) < epsilon:
            print('Found solution for x after',n,'iterations.')
            return print(f"The value of x,y respectively is: {x,y}")
        if determinant(x,y) == 0:
            print('Zero derivative. No solution found.')
            return None
        x = x - f(x,y)/determinant(x,y)
        y = y - g(x,y)/determinant(x,y)
    print('Exceeded maximum iterations. No solution found.')
    return None

start = timeit.default_timer()
solution = newton(f(x,y),g(x,y),determinant(x,y),x,y,epsilon,max_iter)
stop = timeit.default_timer()
print('Time of changed: ', stop - start)  

plt.plot(values_x,values_y, linestyle ="",marker="x")
plt.ylim([0,4])
plt.xlim([0,3])
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.show()