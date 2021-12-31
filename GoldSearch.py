# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 12:38:15 2021

@author: Oscar
"""

# Golden Search, mimics the secant method, but for finding the Global Max and min (optimization of a function)
# Strategy in selecting the bounds of the interval:
    # l0 = distance between estimate,
    # l0 = l1+l2 ; l1/l0 = l2/l1
    # R = (l2/l1)**-1 (reciprocal)
    # From substitution : 1 +R = 1/R -> R**2 + R - 1 = 0
    # R = [sqrt(5)-1]/2 <- GOLDEN RATIO
        # d = R(x_u - x_l)
        #x1 = x_l + d ; x2 = x_u - d 
        
import numpy as np
import math
import matplotlib.pyplot as plt
        
"""
Interval Selection
"""
# Parameters
xu = 20 #int(input("Please choose a upper bound: "))
xl = -20 #int(input("Please choose a lower bound: "))
N = 100 #int(input("Please choose Maxt number of iterations: "))
# Golden Ratio
R = (math.sqrt(5) - 1)/2


"""
Evaluation of the Function
"""
# Evaluated function
f = lambda x: 2*np.sin(x) - x**2/10

def GoldenSearchMax(xu, xl, f, N):
        
    for i in range(0, N-1):
        # Intermediate points
        d = R*(xu - xl)
        
        x1 = xl + d
        x2 = xu - d
        
        fx1, fx2 = f(x1), f(x2)
        
        if fx1 > fx2 :
            xl = x2
            
        elif fx1 < fx2:
            xu = x1
            
        else:
            #print("The local maxima is located at:", x1, fx1)
            break
    return x1, fx1

def GoldenSearchMin(xu, xl, f, N):
        
    for i in range(0, N-1):
        # Intermediate points
        d = R*(xu - xl)
        
        x1 = xl + d
        x2 = xu - d
        
        fx1, fx2 = f(x1), f(x2)
        
        if fx1 < fx2 :
            xl = x2
            
        elif fx1 > fx2:
            xu = x1
            
        else:
            #print("The local minima is located at:", x1, fx1)
            break
    return x1, fx1

# Arrays to store the numbers
Max = GoldenSearchMax(xu, xl, f, N)
Min = GoldenSearchMin(xu, xl, f, N)
print('The local max and min of the interval is:', Max, Min)

# Initializing Arrays
x_value = np.linspace(xl, xu, N-1)
y_value = np.zeros(N-1)

# Populating y_array
for k in range(N-1):
    y_value[k] = f(x_value[k])

# Plotting the function f
plt.plot(x_value ,y_value)
plt.scatter(Max[0], Max[1], label = 'Maxima', color = 'r')
plt.scatter(Min[0], Min[1], label = 'Maxima', color = 'g')
plt.legend(['Function', 'Maxima', 'Minima'])
plt.xlabel('x')
plt.ylabel('y')
plt.show()
