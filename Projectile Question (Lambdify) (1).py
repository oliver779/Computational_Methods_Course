# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 23:48:48 2021

@author: Oscar
"""

# Tutorial 9: Trajectory of a thrown object

#Import Libraries
import numpy as np
import sympy as sp
import math as m
import matplotlib.pyplot as plt

# Define Parameters

# Initial Guess
#Min root
x_min0 = -3

#Max Root
x_max0 = 56

# Tolerance
Tolerance = 1e-10

# Initial Thrown angle
a0 = m.radians(30) 

# Initial Velocity
v0 = 25 #(m/s)

# Initial Height
y0 = 1.5 #(m)

# Gravitational Accel
g = 9.81 #(m/s**2)

# Maxiterations
Max_iter = 100

# Define the Actual Equation
x = sp.Symbol('x')

func = sp.tan(a0)*x - (g/ (2 * v0**2 * (sp.cos(a0))**2) * x**2) + y0
f = sp.lambdify(x, func)

func_prime = sp.diff(func, x)
f_prime = sp.lambdify(x, func_prime)

func_prime2 = sp.diff(func, x, 2)
f_prime2 = sp.lambdify(x, func_prime2)

#%% Question A: roots
def NewtonRapshon(f, f_prime, x0, Tolerance, Max_iter):
    
    xn = x0
    
    for i in range(0,Max_iter):
        # If the value of the function at x is less than the tolerance, we print xn as it is close to the root
        if abs(f(xn)) < Tolerance:
            #print('Found solution after',i,'iterations.')
            return xn
        # Find the derivative at the value of xn, if none exists
        if f_prime(xn) == 0:
            #print('Zero derivative. No solution found.')
            return None
        # Calculate the new value of xn and repeat the process
        xn = xn - f(xn)/f_prime(xn)
    # If the Max number of iterations are reached, stop the program  
    print('Exceeded maximum iterations. No solution found.')
    return 

Min_root = NewtonRapshon(f, f_prime, x_min0, Tolerance, Max_iter)
Max_root = NewtonRapshon(f, f_prime, x_max0, Tolerance, Max_iter)
print('The ball will travel a distance of %.3f m' %Max_root)

#%% Question B: optimization
def NewtonOptimize(f_prime, f_prime2, x0, Tolerance, Max_iter):
    
    x_i = x0
    
    for i in range(0,Max_iter):
        xn = x_i - f_prime(x_i) / f_prime2(x_i)
        # If the value of the function at x is less than the tolerance, we print xn as it is close to the root
        if abs(f_prime(x_i)) < Tolerance:
            #print('Found solution after',i,'iterations.')
            return xn, f(xn)
        # Find the derivative at the value of xn, if none exists
        if np.float(f_prime(x_i)) == 0:
            #print('Solution found at', x_i)
            return None
        # Calculate the new value of xn and repeat the process
        x_i = xn
    # If the Max number of iterations are reached, stop the program  
    print('Exceeded maximum iterations. No solution found.')
    return 

max_distance, max_height = NewtonOptimize(f_prime, f_prime2, x_max0, Tolerance, Max_iter)
print('The Max height is achieved at x= %.3f m' %max_distance , 'reaching a height of y= %.3f m' %max_height)

#%% Question 3: Sensitivity to angle change
a0 = m.radians(29)
func = sp.tan(a0)*x - (g/ (2 * v0**2 * (sp.cos(a0))**2) * x**2) + y0
solutionDown = NewtonRapshon(f, f_prime, x_max0, Tolerance, Max_iter)
print('The ball will travel a distance of %.3f m with an angle %.3f' %(solutionDown,a0*180/np.pi))

a0 = m.radians(30)
func = sp.tan(a0)*x - (g/ (2 * v0**2 * (sp.cos(a0))**2) * x**2) + y0
solutionOld = NewtonRapshon(f, f_prime, x_max0, Tolerance, Max_iter)
print('The ball will travel a distance of %.3f m with an angle %.3f' %(solutionOld,a0*180/np.pi))

a0 = m.radians(31)
func = sp.tan(a0)*x - (g/ (2 * v0**2 * (sp.cos(a0))**2) * x**2) + y0
solutionUp = NewtonRapshon(f, f_prime, x_max0, Tolerance, Max_iter)
print('The ball will travel a distance of %.3f m with an angle %.3f' %(solutionUp,a0*180/np.pi))

sensitivityUp=abs((solutionUp-solutionOld)/solutionOld)
sensitivityDown=abs((solutionDown-solutionOld)/solutionOld)

print(f'A 1° increase at a launch angle of 30° gives a sensitivity of {sensitivityUp:.2f}metre per degree increase of the launch angle')
print(f'A 1° decrease at a launch angle of 30° gives a sensitivity of {sensitivityDown:.2f}metre per degree decrease of the launch angle')

#%% Question 4: if xf = 50, what is the optimum angle and pitch velocity

# Defining desired results
xf = 50

# Define Angles
angles = np.linspace(89.9, 0, 10000)

# Defining Functions
v = sp.Symbol('v')
a = sp.Symbol('a') # theta
vel = sp.tan(a)*xf - (g/ (2 * v**2 * (sp.cos(a))**2) * xf**2) + y0
f = sp.lambdify([v,a], vel)
vel_prime = sp.diff(vel, v)
v_prime = sp.lambdify([v,a], vel_prime)

velocity=[]
theta = np.zeros(len(angles))

for i in range(len(angles)):
    
    theta[i]=np.radians(angles[i])
    
    def NewtonVel(f, v_prime, v0, theta, Tolerance, Max_iter):
        
        vn = v0
        
        for n in range(0,Max_iter):
            # If the value of the function at x is less than the tolerance, we print xn as it is close to the root
            if abs(f(vn,theta[i])) < Tolerance:
                #print('Found solution after',i,'iterations.')
                return vn
            # Find the derivative at the value of xn, if none exists
            if v_prime(vn, theta[i]) == 0:
                #print('Zero derivative. No solution found.')
                return None
            # Calculate the new value of xn and repeat the process
            vn = vn - f(vn,theta[i])/v_prime(vn, theta[i])
        # If the Max number of iterations are reached, stop the program  
        print('Exceeded maximum iterations. No solution found.')
        return vn
    
    velocity.append(NewtonVel(f, v_prime, 15, theta, Tolerance, Max_iter))

#%% Plot range of solutions
plt.plot(angles,velocity)
plt.ylabel('Launch Velocity [m/s]')
plt.xlabel('Launch Angle [Degrees]')
plt.ylim([10,40])
plt.xlim([0,90])
Opt_Angle=angles[velocity.index(min(velocity))]
a=min(velocity)
print(f'The minimum launch velocity to achieve a distance of 50 m is {a:.2f} m/s at an angle of {Opt_Angle:.2f} degrees')
