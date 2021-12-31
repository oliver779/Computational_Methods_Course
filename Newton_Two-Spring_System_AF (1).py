# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 17:43:51 2020

@author: emc1977
"""
# Energy of the system can be found as 
# 1/2*kb*(sqrt(x1^2+(Lb+x2)^2)-Lb)^2
# 1/2*ka*(sqrt(x1^2+(La-x2)^2)-La)^2
# -F1x1-F2x2
import sympy
import numpy as np
x,y = sympy.symbols('x,y')

#need the following to create functions out of symbolix expressions
from sympy.utilities.lambdify import lambdify
from sympy import symbols, Matrix, Function, simplify, exp, hessian, solve, init_printing
init_printing()

ka=9.
kb=2.
La=10.
Lb=10.
F1=2.
F2=4.

X = Matrix([x,y])

f = Matrix([0.5*(ka*((x**2+(La-y)**2)**0.5 - La)**2)+0.5*(kb*((x**2+(Lb+y)**2)**0.5 - Lb)**2)-F1*x-F2*y])
print(np.shape(f))

#Since the Hessian is 2x2, then the Jacobian should be 2x1 (for the matrix multiplication)
gradf = simplify(f.jacobian(X)).transpose()
# #Create function that will take the values of x, y and return a jacobian
# #matrix with values
fgradf = lambdify([x,y], gradf)
print('Jacobian f', gradf)

hessianf = simplify(hessian(f, X))
# #Create a function that will return a Jessian matrix with values
fhessianf = lambdify([x,y], hessianf)
print('Hessian f', hessianf)


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
        print(X)

        X_prev = X
        #X had dimensions (2,) while the 2nd term (2,1), so it had to be converted to 1D
        X = X - np.matmul(np.linalg.inv(Hess(x,y)), Grad(x,y)).flatten()
        error = X - X_prev
        x,y = X[0], X[1]

    return X, iter_x,iter_y, iter_count

root,iter_x,iter_y, iter_count = Newton_Raphson_Optimize(fgradf,fhessianf,1,1)
print(root)
