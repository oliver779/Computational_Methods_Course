# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 10:47:13 2020

@author: chris
"""
import matplotlib.pyplot as plt

# import all modules . Not that I commented out the f definition within modules 
# ex1 to ex4. This is so I can define f in the master script here.
# If f is defined for example within ex1 this will take priority as it is located
# within the ex1 module and therefore if I redefine f in this master script it won't
# calculate the integral for this new function except if I also change it in ex1.py
# 
# The same applies for a,b and n which I did not define within each module

import ex1    
import ex2
import ex3
import ex4

# define function here
def f(x):
    return x**2+4*x-12

#analytical integral used to obtain exact integral value
def Integral_f(x):
    return 1/3*x**3+2*x**2-12*x

   

a=-10  #lower bound
b=10  #upper bound

n=1000 #maximum amount of subdivisions used in the loop at the bottom

Exact_Integral=Integral_f(b)-Integral_f(a)  #exact integral value used for error control

# I have defined all my variables as empty lists so that I can add a value within
# the loop that follows using the append method (read about it online).
# An alternative method would have been to use numpy and np.zeros(n) to get a
# vector of the correct lenght and replacing the zeros within the loop at the bottom

rect=[]     
rect_error=[]

trapezoid=[]
trapezoid_error=[]

simpson=[]
simpson_error=[]

simpson38=[]
simpson38_error=[]

x=[]

#Note that the way we defined simpson's rule we need to use an even number of subdivisions
# That is why I am starting with 2 subdivisions and increasing in steps of 2

for n in range(2,n,2):
    
    x.append(n)
    rect.append(ex1.rect_rule(f, a,b, n))  # using the rect_rule function defined in the ex1.py module
    # and adding it to the rect array
    
    rect_error.append(abs(ex1.rect_rule(f, a,b, n)-Exact_Integral)) #calculating the absolute
    # value of the error and adding it to the rect_error array
    
    trapezoid.append(ex2.trapz(f,a,b,n))
    trapezoid_error.append(abs(ex2.trapz(f, a,b, n)-Exact_Integral))
    
    simpson.append(ex3.simps(f,a,b,n))
    simpson_error.append(abs(ex3.simps(f, a,b, n)-Exact_Integral))
    
    simpson38.append(ex4.calculate(f,a,b,n))
    simpson38_error.append(abs(ex4.calculate(f,a,b,n)-Exact_Integral))
    
#%%
#plotting the solution for the integral as a function of subdivisions used
plt.plot(x,rect,x,trapezoid,x,simpson,x,simpson38)
plt.ylim(400,450)
plt.legend(['MidPoint Rule','Trapezoidal','Simpson Rule','3/8 Simpson'])
plt.show()

#plotting the error as a function of steps used
plt.semilogy(x,rect_error,x,trapezoid_error,x,simpson_error)
plt.legend(['MidPoint Rule','Trapezoidal','Simpson Rule'])


