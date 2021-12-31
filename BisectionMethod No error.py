# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 11:10:06 2021

@author: Oscar
"""

#Bisection Method
#Choosing lower bound and upper bound x-values such that the function changes signs across the interval
    #aka F(x_l)*F(x_u) < 0
#An estimate of the root can be found as Xr=(x_l + X_u)/2
#Determine the subinterval which the root lies in
    #a) if F(x_L)F(x_R)<0 -> real root would lie between [x_l, x_r]
    #b) if F(x_L)F(x_R)>0 -> real root would lie between [x_r, x_u]
    #c) if F(x_L)F(x_R)=0 X_r is the root
"""
Bisection Method Attempt
"""
#Def the section on the perimeters
def Bisection(F,a,b,n):
    # F would be the function for which we're attempting
    # a would be lower bound x
    # b would be upper bound x
    # N is the number of iterations required
    
    if F(a)*F(b) >= 0: #Choosing lower bound and upper bound x-values such that the function changes signs across the interval
        print("Bounds choosen does not meet the requirement of the Bisection method")
        return None
        
    # Iterative Solution: aka For loop
    for i in range(1, n+1):
        
        c = (a + b)/2 # c = estimate root
        
        if F(a)*F(c) < 0: # If true, root lies with in <x_L, x_r>, therefore repeat step 2 with new x_u = x_r
            a = a
            b = c

        elif F(b)*F(c) < 0: # If true, root lies with in <x_r, x_U>, therefore repeat step 2 with new x_L = x_r
            a = c
            b = b
            
        elif F(a)*F(c) == 0: # Solution is found
            print("Found exact solution.")
            return c
        
        else:
            print("Bisection method fails.")
            return None
    return (a + b)/2

F = lambda x: x**2 + 4*x - 12 # creates the function, where "lambda" as a function of x
tol = 1e-6
approx_phi = Bisection(F, -5, 5, 25)
print(" A root of the function is found at x = ", approx_phi)
"""
#Requires Error control, else will continue to infinity
    # True Error, E_t = True Value - imperfect value [Absolute Error = True - Measured]
    # Fractional Relative Error E_f= True Error/ True Value [Relative Error = AbsE/Value ]
    # Approx Fractional Error E_a = (current approx - previous approx)/current approx
        #for |E_a| to be acceptable, E_a < E_s: E_s is a pre-selected acceptable error
            #In practice?
                #Define Es
                #Choose function of interest f(x)
                #Choose a value of x to evalutate f(x)
                #Express function in form for numerical calculation: ie a Series
                #Calc with the number of series terms required to satisfy Es
"""