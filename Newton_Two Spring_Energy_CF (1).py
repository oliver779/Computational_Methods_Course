# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 17:38:32 2020

@author: chris
"""
import time
import sympy
import numpy as np
xx,yy = sympy.symbols('x[0],x[1]')  #I replaced the symbolics to x[0] and x[1] 
#so I can directly copy past this into the jacobian function below without 
#having to change the variable names

from sympy import symbols, Matrix,Function, simplify, exp, hessian, solve

ka=9
kb=2
La=10
Lb=10
F1=2
F2=4

X = Matrix([xx,yy])
g = Matrix([0.5*(ka*((xx**2+(La-yy)**2)**0.5 - La)**2)+0.5*(kb*((xx**2+(Lb+yy)**2)**0.5 - Lb)**2)-F1*xx-F2*yy])

gradf = simplify(g.jacobian(X))
print('Jacobian g', gradf)

hessianf = simplify(hessian(g, X))
print('Hessian g', hessianf)

from scipy import optimize
import numpy as np

def f(x): # The Spring Energy Minimisation Equation
    return 0.5*(ka*((x[0]**2+(La-x[1])**2)**0.5 - La)**2)+0.5*(kb*((x[0]**2+(Lb+x[1])**2)**0.5 - Lb)**2)-F1*x[0]-F2*x[1]

def jacobian(x): 
    return np.array((-90.0*x[0]*(x[0]**2 + (x[1] - 10)**2)**(-0.5) - 20.0*x[0]*(x[0]**2 + (x[1] + 10)**2)**(-0.5) + 11.0*x[0] - 2, -90.0*x[1]*(x[0]**2 + (x[1] - 10)**2)**(-0.5) - 20.0*x[1]*(x[0]**2 + (x[1] + 10)**2)**(-0.5) + 11.0*x[1] + 900.0*(x[0]**2 + (x[1] - 10)**2)**(-0.5) - 200.0*(x[0]**2 + (x[1] + 10)**2)**(-0.5) - 74.0))

start=time.time()

for i in range(10000):
    solution = optimize.minimize(f, [1,1], method="Newton-CG",jac=jacobian) 
#print(solution,'\n','\n')
end=time.time()

print(solution,'\n',f"Runtime of the program is {end - start}",'\n')

def hessian(x):
    return np.array(((90.0*x[0]**2*(x[0]**2 + x[1]**2 - 20*x[1] + 100)**(-1.5) + 20.0*x[0]**2*(x[0]**2 + x[1]**2 + 20*x[1] + 100)**(-1.5) - 90.0*(x[0]**2 + x[1]**2 - 20*x[1] + 100)**(-0.5) - 20.0*(x[0]**2 + x[1]**2 + 20*x[1] + 100)**(-0.5) + 11.0, 90.0*x[0]*x[1]*(x[0]**2 + x[1]**2 - 20*x[1] + 100)**(-1.5) + 20.0*x[0]*x[1]*(x[0]**2 + x[1]**2 + 20*x[1] + 100)**(-1.5) - 900.0*x[0]*(x[0]**2 + x[1]**2 - 20*x[1] + 100)**(-1.5) + 200.0*x[0]*(x[0]**2 + x[1]**2 + 20*x[1] + 100)**(-1.5)), (90.0*x[0]*x[1]*(x[0]**2 + x[1]**2 - 20*x[1] + 100)**(-1.5) + 20.0*x[0]*x[1]*(x[0]**2 + x[1]**2 + 20*x[1] + 100)**(-1.5) - 900.0*x[0]*(x[0]**2 + x[1]**2 - 20*x[1] + 100)**(-1.5) + 200.0*x[0]*(x[0]**2 + x[1]**2 + 20*x[1] + 100)**(-1.5), 9000.0*(x[0]**2 + (x[1] - 10)**2)**(-1.5)*(0.1*x[1] - 1)**2 - 90.0*(x[0]**2 + (x[1] - 10)**2)**(-0.5) + 2000.0*(x[0]**2 + (x[1] + 10)**2)**(-1.5)*(0.1*x[1] + 1)**2 - 20.0*(x[0]**2 + (x[1] + 10)**2)**(-0.5) + 11.0)))

start=time.time()
for i in range(10000):
    solution1=optimize.minimize(f, [1,1], method="Newton-CG",jac=jacobian,hess=hessian)  
end=time.time()
print(solution1, '\n',f"Runtime of the program is {end - start}",'\n')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
#%% Other option with CG method
start=time.time()
for i in range(10000):
    solution2 = optimize.minimize(f, [1,1], method="CG") 
end=time.time()
print(solution2,'\n',f"Runtime of the program is {end - start}",'\n')

start=time.time()
for i in range(10000):
    solution3 = optimize.minimize(f, [1,1], method="CG",jac=jacobian) 
end=time.time()
print(solution3,'\n',f"Runtime of the program is {end - start}",'\n')