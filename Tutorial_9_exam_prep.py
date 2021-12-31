
#%% PART 1
import numpy as np
import matplotlib.pyplot as plt
import timeit
import math

y_0 = 1.5
theta = np.radians(13.79483459)
g = 9.81
v_0 = 30.72081747
x = np.linspace(0,70,1000)

def y(x):
    return (np.tan(theta))*x - (g)/(2*v_0**2*(np.cos(theta))**2)*x**2 + y_0

plt.plot(x, y(x))
plt.axhline(0)

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

f = lambda x: (np.tan(theta))*x - (g)/(2*v_0**2*(np.cos(theta))**2)*x**2 + y_0
Df= lambda x: (np.tan(theta)) - 2*(g)/(2*v_0**2*(np.cos(theta))**2)*x
x = 50
epsilon=0.01
max_iter=1000
start = timeit.default_timer()
solution = newton(f,Df,x,epsilon,max_iter)
stop = timeit.default_timer()
print('Time of changed: ', stop - start)  
print(f'The position at which the projectile hits the ground is at x: {solution}')

#%% PART 2
# Using Newton Raphson again to find the maximum height reached

def fprime(x):
    return (np.tan(theta)) - 2*(g)/(2*v_0**2*(np.cos(theta))**2)*x
def fsecond(x):
    return  - 2*(g)/(2*v_0**2*(np.cos(theta))**2)
def quadratic_approx(x, x0, f, fprime, fsecond):
    return f(x0)+fprime(x0)*(x-x0)+0.5*fsecond(x0)*(x-x0)**2
def newton2(x0, fprime, fsecond, maxiter=100, eps=0.0001):
    x=x0
    for i in range(maxiter):
        xnew=x-(fprime(x)/fsecond(x))
        if xnew-x<eps:
            return xnew
            print('converged')
            break
        x = xnew
    return x

x_star=newton2(0, fprime, fsecond)
print(f'The maximum height is reached at x: {x_star}')

#%%PART 3
# Finding the sensitivity of the landing spot

theta = np.radians(31)
solution2 = newton(f,Df,x,epsilon,max_iter)
print(f'The position at which the projectile hits the ground is at x: {solution2} and theta: {theta}')
theta2 = theta

theta = np.radians(30)
solution3 = newton(f,Df,x,epsilon,max_iter)
print(f'The position at which the projectile hits the ground is at x: {solution3} and theta: {theta}')
theta3 = theta

theta = np.radians(29)
solution4 = newton(f,Df,x,epsilon,max_iter)
theta4 = theta
print(f'The position at which the projectile hits the ground is at x : {solution4} and theta: {theta}')
print(f'An average change of theta: {(theta2-theta3+theta3-theta4)/2} equals to a change in height of: {(solution3-solution4 + solution2-solution3)/2}')

#%%PART 4
# Optimum angle and initial pitch velocity to get the projectile to hit the ground to 50 m

# DOING IT IN EXCEL



