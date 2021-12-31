# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 20:14:10 2021

@author: Oliver
"""

"""Univariate Optimisation"""

# Golden Search Technique for finding the Maximum

import numpy as np

def gsection(ftn, xl, xm, xr, tol = 1e-9):
    # applies the golden-section algorithm to maximise ftn
    # we assume that ftn is a function of a single variable
    # and that x.l < x.m < x.r and ftn(x.l), ftn(x.r) <= ftn(x.m)
    #
    # the algorithm iteratively refines x.l, x.r, and x.m and
    # terminates when x.r - x.l <= tol, then returns x.m
    # golden ratio plus one
    gr1 = 1 + (1 + np.sqrt(5))/2
    #
    # successively refine x.l, x.r, and x.m
    fl = ftn(xl)
    fr = ftn(xr)
    fm = ftn(xm)
    while ((xr - xl) > tol):
        if ((xr - xm) > (xm - xl)):
            y = xm + (xr - xm)/gr1
            fy = ftn(y)
            if (fy >= fm):
                xl = xm
                fl = fm
                xm = y
                fm = fy
            else:
                xr = y
                fr = fy
        else:
            y = xm - (xm - xl)/gr1
            fy = ftn(y)
            if (fy >= fm):
                xr = xm
                fr = fm
                xm = y
                fm = fy
            else:
                xl = y
                fl = fy
    return(xm)
    
xl=0
xm=5
xr=10
def ftn(x):
    return x**2+5*x+10
print(gsection(ftn, xl, xm, xr, tol = 1e-9))
print(ftn(xm))







# Newton Maximizing Algorithm
import math
import matplotlib.pyplot as plt

def f(x):
    return x**3-6*x**2+4*x+2
x = np.linspace(-1, 1)
fig, ax = plt.subplots()
ax.plot(x, f(x), label='target')
ax.grid()

def fprime(x):
    return 3*x**2-12*x+4
def fsecond(x):
    return 6*x - 12
def quadratic_approx(x, x0, f, fprime, fsecond):
    return f(x0)+fprime(x0)*(x-x0)+0.5*fsecond(x0)*(x-x0)**2

x = np.linspace(-1, 1)
fig, ax = plt.subplots()
ax.plot(x, f(x), label='target')
ax.grid()
ax.plot(x, quadratic_approx(x, 0, f, fprime, fsecond), color='red', label='quadratic approximation')
ax.set_ylim([-2,3])
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
plt.legend()

def newton(x0, fprime, fsecond, maxiter=100, eps=0.0001):
    x=x0
    for i in range(maxiter):
        xnew=x-(fprime(x)/fsecond(x))
        if xnew-x<eps:
            return xnew
            print('converged')
            break
        x = xnew
    return x

x_star=newton(0, fprime, fsecond)
fig, ax = plt.subplots()
ax.plot(x, f(x), label='target')
ax.grid()
ax.plot(x, quadratic_approx(x, x_star , f, fprime, fsecond), color='red', label='quadratic approximation')
ax.set_ylim([-2,3])
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
ax.axvline(x = x_star, color='green')
plt.legend()






#Valentina Alto: See: https://medium.com/swlh/optimization-algorithms-the-newton-method-4bc6728fb3b6

# Newton Minimizing Algorithm
def f(x):
    return -x**3+6*x**2-4*x-2
x = np.linspace(-1, 1)
fig, ax = plt.subplots()
ax.plot(x, f(x), label='target')
ax.grid()

def fprime(x):
    return -3*x**2+12*x-4
def fsecond(x):
    return -6*x + 12
def quadratic_approx(x, x0, f, fprime, fsecond):
    return f(x0)+fprime(x0)*(x-x0)+0.5*fsecond(x0)*(x-x0)**2

x = np.linspace(-1, 1)
fig, ax = plt.subplots()
ax.plot(x, f(x), label='target')
ax.grid()
ax.plot(x, quadratic_approx(x, 0, f, fprime, fsecond), color='red', label='quadratic approximation')
ax.set_ylim([-5,3])
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
plt.legend()

def newton(x0, fprime, fsecond, maxiter=100, eps=0.0001):
    x=x0
    for i in range(maxiter):
        xnew=x-(fprime(x)/fsecond(x))
        if xnew-x<eps:
            return xnew
            print('converged')
            break
        x = xnew
    return x

x_star=newton(0, fprime, fsecond)
fig, ax = plt.subplots()
ax.plot(x, f(x), label='target')
ax.grid()
ax.plot(x, quadratic_approx(x, x_star , f, fprime, fsecond), color='red', label='quadratic approximation')
ax.set_ylim([-5,3])
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
ax.axvline(x = x_star, color='green')
plt.legend()

#Valentina Alto: See: https://medium.com/swlh/optimization-algorithms-the-newton-method-4bc6728fb3b6






# Gradient Descent vs Newton Unconstrained Multivariate Optimisation

import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np
from mpl_toolkits import mplot3d

def Rosenbrock(x,y):
    return (1 + x)**2 + 100*(y - x**2)**2

def Grad_Rosenbrock(x,y):
    g1 = -400*x*y + 400*x**3 + 2*x -2
    g2 = 200*y -200*x**2
    return np.array([g1,g2])

def Hessian_Rosenbrock(x,y):
    h11 = -400*y + 1200*x**2 + 2
    h12 = -400 * x
    h21 = -400 * x
    h22 = 200
    return np.array([[h11,h12],[h21,h22]])

def Gradient_Descent(Grad,x,y, gamma = 0.00125, epsilon=0.0001, nMax = 10000 ):
    #Initialization
    i = 0
    iter_x, iter_y, iter_count = np.empty(0),np.empty(0), np.empty(0)
    error = 10
    X = np.array([x,y])
    
    #Looping as long as error is greater than epsilon
    while np.linalg.norm(error) > epsilon and i < nMax:
        i +=1
        iter_x = np.append(iter_x,x)
        iter_y = np.append(iter_y,y)
        iter_count = np.append(iter_count ,i)   
        #print(X) 
        
        X_prev = X
        X = X - gamma * Grad(x,y)
        error = X - X_prev
        x,y = X[0], X[1]
          
    print(X)
    return X, iter_x,iter_y, iter_count


root,iter_x,iter_y, iter_count = Gradient_Descent(Grad_Rosenbrock,-2,2)

#PLOTTING THE SOLUTION

x = np.linspace(-2,2,250)
y = np.linspace(-1,3,250)
X, Y = np.meshgrid(x, y)
Z = Rosenbrock(X, Y)

#Angles needed for quiver plot
anglesx = iter_x[1:] - iter_x[:-1]
anglesy = iter_y[1:] - iter_y[:-1]


#matplotlib inline
fig = plt.figure(figsize = (16,8))

#Surface plot
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(X,Y,Z,rstride = 5, cstride = 5, cmap = 'jet', alpha = .4, edgecolor = 'none' )
ax.plot(iter_x,iter_y, Rosenbrock(iter_x,iter_y),color = 'r', marker = '*', alpha = .4)

ax.view_init(45, 280)
ax.set_xlabel('x')
ax.set_ylabel('y')


#Contour plot
ax = fig.add_subplot(1, 2, 2)
ax.contour(X,Y,Z, 50, cmap = 'jet')
#Plotting the iterations and intermediate values
ax.scatter(iter_x,iter_y,color = 'r', marker = '*')
ax.quiver(iter_x[:-1], iter_y[:-1], anglesx, anglesy, scale_units = 'xy', angles = 'xy', scale = 1, color = 'r', alpha = .3)
ax.set_title('Gradient Descent with {} iterations'.format(len(iter_count)))


plt.show()

#Newton's Method (Multi-Dimensional)

def Newton_Raphson_Optimize(Grad, Hess, x,y, epsilon=0.000001, nMax = 200):
    #Initialization
    i = 0
    iter_x, iter_y, iter_count = np.empty(0),np.empty(0), np.empty(0)
    error = 10
    X = np.array([x,y])
    
    #Looping as long as error is greater than epsilon
    while np.linalg.norm(error) > epsilon and i < nMax:
        i +=1
        iter_x = np.append(iter_x,x)
        iter_y = np.append(iter_y,y)
        iter_count = np.append(iter_count ,i)   
        print(f'X: {X}') 
        
        X_prev = X
        X = X - np.linalg.inv(Hess(x,y)) @ Grad(x,y)
        error = X - X_prev
        x,y = X[0], X[1]
          
    return X, iter_x,iter_y, iter_count


root,iter_x,iter_y, iter_count = Newton_Raphson_Optimize(Grad_Rosenbrock,Hessian_Rosenbrock,-2,2)

x = np.linspace(-3,3,250)
y = np.linspace(-9,8,350)
X, Y = np.meshgrid(x, y)
Z = Rosenbrock(X, Y)

#Angles needed for quiver plot
anglesx = iter_x[1:] - iter_x[:-1]
anglesy = iter_y[1:] - iter_y[:-1]

#matplotlib inline
fig = plt.figure(figsize = (16,8))

#Surface plot
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(X,Y,Z,rstride = 5, cstride = 5, cmap = 'jet', alpha = .4, edgecolor = 'none' )
ax.plot(iter_x,iter_y, Rosenbrock(iter_x,iter_y),color = 'r', marker = '*', alpha = .4)

#Rotate the initialization to help viewing the graph
ax.view_init(45, 280)
ax.set_xlabel('x')
ax.set_ylabel('y')

#Contour plot
ax = fig.add_subplot(1, 2, 2)
ax.contour(X,Y,Z, 60, cmap = 'jet')
#Plotting the iterations and intermediate values
ax.scatter(iter_x,iter_y,color = 'r', marker = '*')
ax.quiver(iter_x[:-1], iter_y[:-1], anglesx, anglesy, scale_units = 'xy', angles = 'xy', scale = 1, color = 'r', alpha = .3)
ax.set_title('Newton method with {} iterations'.format(len(iter_count)))

plt.show()


































