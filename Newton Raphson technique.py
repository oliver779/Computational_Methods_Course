# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 00:24:22 2021

@author: Oliver
"""
import timeit


def newton(f,df,x,epsilon,max_iter):
    for n in range(0,max_iter):
        if abs(f(x)) < epsilon:
            print('Found solution after',n,'iterations.')
            return x
        if Df(x) == 0:
            print('Zero derivative. No solution found.')
            return None
        x = x - f(x)/Df(x)
    print('Exceeded maximum iterations. No solution found.')
    return None
f = lambda x: x**2 + 4*x - 12
Df= lambda x: 2*x + 4 
x=10
epsilon=0.000000000000001
max_iter=100
start = timeit.default_timer()
solution = newton(f,Df,x,epsilon,max_iter)
stop = timeit.default_timer()
print('Time of changed: ', stop - start)  
print(solution)

def newton2(f,df,x,epsilon,max_iter):
    xn = x0
    for n in range(0,max_iter):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            print('Found solution after',n,'iterations.')
            return xn
        Dfxn = Df(xn)
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
    print('Exceeded maximum iterations. No solution found.')
    return None 
x0=10
start2 = timeit.default_timer()
solution2 = newton2(f,Df,x0,epsilon,max_iter)
stop2 = timeit.default_timer()
print('Time of original newton: ', stop2 - start2) 
print(solution2)
Difference = ((stop-start)/(stop2-start2))*100
print(f'The difference between original and changed in time is {Difference} %')
