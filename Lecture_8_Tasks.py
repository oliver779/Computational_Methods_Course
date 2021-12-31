# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 13:39:48 2021

@author: Oliver

"""


"""Roots of Equation Systems"""
import math

import numpy as np

"""Finding roots of multi-variable equations"""

x = 1.5
y = 3.5

dudx = lambda x, y: 2*x+y
dvdx = lambda y: 3*y**2
dudy = lambda x: x
dvdy = lambda x, y: 1+6*x*y

a = np.array([[dudx(x,y),dudy(x)], [dvdx(y),dvdy(x,y)]]) # Jacobian Matrix

initial_u_guess = lambda x, y: x**2 + x*y - 10
initial_v_guess = lambda x, y: y + 3*x*y**2 -57

x_new = x - ((initial_u_guess(x, y)*dvdy(x,y) - initial_v_guess(x, y)*x))/np.linalg.det(a)
y_new = y - (initial_v_guess(x, y)*dudx(x,y) - (initial_u_guess(x, y)*dvdx(y)))/np.linalg.det(a)
x_sol = 2
y_sol = 3
counter = 0
error = 0

while True:
    counter += 1
    x_new = x - ((initial_u_guess(x, y)*dvdy(x,y) - initial_v_guess(x, y)*x))/np.linalg.det(a)
    y_new = y - (initial_v_guess(x, y)*dudx(x,y) - (initial_u_guess(x, y)*dvdx(y)))/np.linalg.det(a)

    if x_sol - error<=x_new<=x_sol + error:
        print(f'The solution has been found after {counter} iterations and the values are x: {x_new} and y: {y_new}')
        break
    if counter == 3:
        print(f'The solution after {counter} iterations is: x: {x_new}, y: {y_new} with relative error in: x: {abs(x_new-x_sol)/x_sol} and y: {abs(y_new-y_sol)/y_sol}')
    else:
        x = x_new
        y = y_new

"""Roots of Polynomial Equations"""
# Factorization of Polynomials

f = [1,2,-24]
guess = 4
values =[f[0]]
z = f[0]
for i in range(0,len(f)-2):
    x = f[i+1] + z*guess
    print(x)
    z = x
    values.append(x)
print(values)

"""Polynomial Division"""

def expanded_synthetic_division(dividend, divisor):
    """Fast polynomial division by using Expanded Synthetic Division. 
    Also works with non-monic polynomials.

    Dividend and divisor are both polynomials, which are here simply lists of coefficients. 
    E.g.: x**2 + 3*x + 5 will be represented as [1, 3, 5]
    """
    out = list(dividend)  # Copy the dividend
    normalizer = divisor[0]
    for i in range(len(dividend) - len(divisor) + 1):
        # For general polynomial division (when polynomials are non-monic),
        # we need to normalize by dividing the coefficient with the divisor's first coefficient
        out[i] /= normalizer

        coef = out[i]
        if coef != 0:  # Useless to multiply if coef is 0
            # In synthetic division, we always skip the first coefficient of the divisor,
            # because it is only used to normalize the dividend coefficients
            for j in range(1, len(divisor)):
                out[i + j] += -divisor[j] * coef

    # The resulting out contains both the quotient and the remainder,
    # the remainder being the size of the divisor (the remainder
    # has necessarily the same degree as the divisor since it is
    # what we couldn't divide from the dividend), so we compute the index
    # where this separation is, and return the quotient and remainder.
    separator = 1 - len(divisor)
    return out[:separator], out[separator:]  # Return quotient, remainder.

if __name__=='__main__':
    print ("POLYNOMINAL SYNTHETIC DIVISION")
    N = [1, 2, -24]
    D = [1, -4]
    print (" %s /%s =" % (N,D),)
    print (" %s remainder %s" % expanded_synthetic_division(N, D))

"""Muller's Technique - Finding Roots"""

def f(x):
    return x**3 - 13*x - 12

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
        print(f'Iteration: {iter}, Value of root: {x_r}, Error: {abs((x2/x_r*100)-100)}%')    
        if (abs(dx_r) < eps * x_r or iter >= maxit):
            break
        x0 = x1
        x1 = x2
        x2 = x_r
    return None
x_r = 20
h = 0.1
eps = 0.01
maxit = 10

solution = Muller(x_r, h, eps, maxit)
print(solution)

