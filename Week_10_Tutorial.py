import math as m
import matplotlib.pyplot as plt
import numpy as np
from sympy import *

epsilon = 0.01
x_min = -10
x_max = 60
x = np.linspace(x_min,x_max,10000)
theta_values = []
    
y_initial = 1.5
v_initial = 25
g = 9.81
x0 = 50
x02 = 0
theta_initial = m.radians(30)
y = m.tan(theta_initial)*x - g/(2*v_initial**2*(m.cos(theta_initial))**2)*x**2+y_initial

x1 = Symbol('x')
f = Function('f')(x)
f = m.tan(theta_initial)*x1 - (g/ (2 * v_initial**2 * m.cos(theta_initial)**2) * x1**2) + y_initial
f_prime = diff(f, x1)
f_prime2 = diff(f, x1, 2)


def y(x): 
    return m.tan(theta_initial)*x - g/(2*v_initial**2*(m.cos(theta_initial))**2)*x**2+y_initial

def y_prime(x):
    return -0.659675690427106*x - 6.40533119664628

def y_prime_squared():
    return-0.659675690427106
    
def quadratic_approx(x):
    return y(x0)+y_prime(x0)*(x-x0)+0.5*y_prime_squared()*(x-x0)**2

y_approx_values = []
y_values = []

for i in x:
    y_approx_values.append(quadratic_approx(i))
    y_values.append(y(i))
    


# for i in x:
#     y_values.append(float(y))
plt.hlines(0,x_min,x_max)
# plt.plot(x,y_approx_values)
plt.plot(x,y_values)
plt.show()


for i in x:
    if -epsilon<=quadratic_approx(i)<=epsilon:
        print(float(quadratic_approx(i)),i)

def newton(f,Df,x0,epsilon,max_iter):
    '''Approximate solution of f(x)=0 by Newton's method.

    Parameters
    ----------
    f : function
        Function for which we are searching for a solution f(x)=0.
    Df : function
        Derivative of f(x).
    x0 : number
        Initial guess for a solution f(x)=0.
    epsilon : number
        Stopping criteria is abs(f(x)) < epsilon.
    max_iter : integer
        Maximum number of iterations of Newton's method.

    Returns
    -------
    xn : number
        Implement Newton's method: compute the linear approximation
        of f(x) at xn and find x intercept by the formula
            x = xn - f(xn)/Df(xn)
        Continue until abs(f(xn)) < epsilon and return xn.
        If Df(xn) == 0, return None. If the number of iterations
        exceeds max_iter, then return None.

    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> Df = lambda x: 2*x - 1
    >>> newton(f,Df,1,1e-8,10)
    Found solution after 5 iterations.
    1.618033988749989
    '''
    xn = x0
    for n in range(0,max_iter):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            # print('Found solution after',n,'iterations.')
            return xn
        Dfxn = Df(xn)
        if Dfxn == 0:
            # print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
    print('Exceeded maximum iterations. No solution found.')
    return None

f = lambda x: m.tan(theta_initial)*x - g/(2*v_initial**2*(m.cos(theta_initial))**2)*x**2+y_initial
df= lambda x: m.tan(theta_initial) - 2*g/(2*v_initial**2*(m.cos(theta_initial))**2)*x
dff = lambda : - 2*g/(2*v_initial**2*(m.cos(theta_initial))**2)

max_iter=10000
solution = newton(f,df,x0,epsilon,max_iter)
solution2 = newton(f,df,x02,epsilon,max_iter)

print(f'The first solution: {solution}')
print(f'The second solution: {solution2}')

"""Finding the maximum value of the graph"""
# set the expression, y, equal to 0 and solve
result1 = solve(f_prime,x1)
print(result1)


"""sensitivity part"""
theta_sensitivity = np.linspace(m.radians(30),m.radians(45),15)
new_x_values =  []
theta_values = []

while theta_initial <=m.radians(45):
    z = newton(f,df,x0,epsilon,max_iter)
    theta_values.append(theta_initial)
    theta_initial += m.radians(1)
    new_x_values.append(z)

n_theta_values = []
n_new_x_values = []
for element in theta_values:
    if element != None and element>0.06:
        n_theta_values.append(element)
        
for element in new_x_values:
    if element != None:
        n_new_x_values.append(element)
    # if element == None:
    #     n_new_x_values.append(0)
        
plt.plot(n_theta_values,n_new_x_values)
plt.show()





