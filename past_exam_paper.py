# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 12:35:14 2021

@author: Oliver
"""
#%% IMPORTING VARIABLES

import matplotlib.pyplot as plt
import math
import numpy as np
import sympy as sp
import timeit

#%% QUESTION 3

current_value_P = 115000
payment_per_year_A = 25500
no_years_n = 6
x = sp.Symbol('x')
y = sp.Function('y')(x)
y = current_value_P*(x*(1+x)**no_years_n)/((1+x)**no_years_n - 1) -payment_per_year_A
dy = sp.diff(y)
print(dy)
z = np.linspace(-0.9,0.9,100)
g = current_value_P*(z*(1+z)**no_years_n)/((1+z)**no_years_n - 1) -payment_per_year_A
plt.ylabel('The value of difference between payment per year and the interest rate')
plt.xlabel('The interest rate')
plt.plot(z,g,'r')
plt.show()

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
            print('Found solution after',n,'iterations.')
            return xn
        Dfxn = Df(xn)
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
    print('Exceeded maximum iterations. No solution found.')
    return None

f = lambda x: current_value_P*(x*(1+x)**no_years_n)/((1+x)**no_years_n - 1) -payment_per_year_A
df= lambda x: -690000*x*(x + 1)**11/((x + 1)**6 - 1)**2 + 690000*x*(x + 1)**5/((x + 1)**6 - 1) + 115000*(x + 1)**6/((x + 1)**6 - 1)
x0=10
epsilon=0.000001
max_iter=100
solution = newton(f,df,x0,epsilon,max_iter)
print(f'The interest rate is : {solution*100}%')
print(f'proof that it works by plugging the value into the function to make it equal to zero: {f(0.08822160280603923)}')

"""FOR THIS PROGRAM, WE USED NEWTONS METHOD FOR CALCULATING THE ZERO OF THE FUNCTION. INITIALLY THE FUNCTION HAD
TO BE TRANSFORMED TO BE EQUAL TO ZERO, BY USING NEWTON, WE CAN QUICKLY AND EFFICIENTLY FIND THE ZERO OF THE FUNCTION
THERE ARE DRAWBACKS TO THIS METHOD AS IT INVOLVES TAKING THE DERIVATIVE OF THE FUNCTIONS HOWEVERE WE USED SYMPY FOR 
THAT IN ORDER TO MAKE LIFE EASIER ONCE WE FOUND THE ZERO, WE PLUGGED IN THE VALUE INTO THE FUNCTION AGAIN TO SEE IF 
IT EQUALS TO ZERO AND FOUND THAT IT IS CLOSE ENOUGH"""

#%% QUESTION 5

t_init = 0

def model3(t):
    g = 9.81
    h = 1
    s = 1
    r = 1
    a = 0.01
    A = 3.14159*(r)**2
    Q = a*math.sqrt(2*g*h)
    Volume = 3.14159*r**2*h
    Volume2 = 3.14159*h*(r**2+h*s)
    dVdt = Volume2 - Q*t
    return dVdt
error = 0.1

t = np.linspace(0,142,1000)
new_volume = []
for i in t:
    new_volume.append(model3(i))
    if -error<= model3(i) <= error:
        print(f'the time at which the volume reaches zero: {i}')

plt.plot(t,new_volume)
plt.show()

""" FOR THIS PROGRAM WE FIRST HAD TO CHECK WHAT THE INITIAL TIME OF DRAINAGE IS FOR JUST THE CYLINDER
WHICH WAS FOUND TO BE AROUND 72 SECONDS, SO WE HAD TO FIND A SLOPE FOR WHICH THE TIME BECOMES 142 SECONDS, BY INITIAL TESTING AND 
SUBSTITUTING 1 FOR S WE FOUND THAT THE VALUE IS 142, THUS THE ANSWER IS THAT S = 1"""
#%% QUESTION 4

u = 1.8*10**3
m_0 = 160*10**3
q = 2.5*10**3
t = np.linspace(0,30,100)
def velocity(t):
    return u*math.log(abs(m_0/(m_0-q*t)))

velocity_values = []

for i in t:
    velocity_values.append(velocity(i))
    
plt.plot(t,velocity_values)
plt.show()

def calculate_dx (a, b, n):
	return (b-a)/float(n)

def rect_rule (f, a, b, n):
	total = 0.0
	dx = calculate_dx(a, b, n)
	for k in range (0, n):
        	total = total + f((a + (k*dx)))
	return dx*total

print(f'value of integral {rect_rule(velocity, 0, 30, 1000)}')

""" THE RECTANGULAR RULE IS PRETTY ACCURATE WITH A LARGE NUMBER OF RECTANGLES AND CAN BE A REALLY GOOD APPROXIMATION 
OF THE INTEGRAL OF A CERTAIN FUNCTION, THERE ARE BETTER WAYS OF DOING THIS EXCERCISE HOWEVER THIS SEEMED THE FASTEST GIVEN THE 
EXAMINATION CIRCUMSTANCES, THE CODE DIVIDES THE UPPER AND LOWER BOUND INTO N RECTANGLES, IT THEN CALCULATES THE AREA OF
THOSE RECTANGLES SEPERATELY AND ADDS THEM TOGETHER. THIS IS OF COURSE THEORETICAL, THE RECTANGULAR RULE HAS AN ERROR WHICH 
BECOMES ZERO AS THE NUMBER OF RECTANGLES APPROACHES INFINITY"""

#%% QUESTION 2

def model2(x,y_1,y_2):
    f_1 = y_2
    f_2 = -y_2*x
    return [f_1,f_2]

def model(x,y_1,y_2):
    f_1 = y_2
    f_2 = -y_2*math.sin(x)
    return [f_1 , f_2]
# ------------------------------------------------------
omega = 0
g = 9.81
l = 9.81
# ------------------------------------------------------
# initial conditions
x0 = 3.14159/4
y0_1 = omega
y0_2 = -g/l
# total solution interval
x_final = 30
# step size
h = 0.001
# ------------------------------------------------------


# ------------------------------------------------------
# Euler method

# number of steps
n_step = math.ceil(x_final/h)

# Definition of arrays to store the solution
y_1_eul = np.zeros(n_step+1)
y_2_eul = np.zeros(n_step+1)
x_eul = np.zeros(n_step+1)

# Definition 2 of arrays to store the solution
y_1_eul2 = np.zeros(n_step+1)
y_2_eul2 = np.zeros(n_step+1)
x_eul2 = np.zeros(n_step+1)

# Initialize first element of solution arrays 
# with initial condition
y_1_eul[0] = y0_1
y_2_eul[0] = y0_2
x_eul[0]   = x0 

# Populate the x array
for i in range(n_step):
    x_eul[i+1]  = x_eul[i]  + h
    if i == 10:
        print(f'At 10:{x_eul[i+1] }')
    if i == 20:
        print(f'At 20: {x_eul[i+1] }')
    if i == 30:
        print(f'At 30: {x_eul[i+1]}')

# Apply Euler method n_step times
for i in range(n_step):
    # compute the slope using the differential equation
    [slope_1 , slope_2] = model(x_eul[i],y_1_eul[i],y_2_eul[i]) 
    # use the Euler method
    y_1_eul[i+1] = y_1_eul[i] + h * slope_1
    y_2_eul[i+1] = y_2_eul[i] + h * slope_2
    if i == 10:
        print(f'At 10:{y_1_eul[i+1],y_2_eul[i+1]}')
    if i == 20:
        print(f'At 20: {y_1_eul[i+1],y_2_eul[i+1]}')
    if i == 30:
        print(f'At 30: {y_1_eul[i+1],y_2_eul[i+1]}')
    # print(y_1_eul[i],y_2_eul[i])

# Apply Euler method n_step times
for i in range(n_step):
    # compute the slope using the differential equation
    [slope_1 , slope_2] = model2(x_eul2[i],y_1_eul2[i],y_2_eul2[i]) 
    # use the Euler method
    y_1_eul2[i+1] = y_1_eul2[i] + h * slope_1
    y_2_eul2[i+1] = y_2_eul2[i] + h * slope_2
    if i == 10:
        print(f'At 10:{y_1_eul2[i+1],y_2_eul2[i+1]}')
    if i == 20:
        print(f'At 20: {y_1_eul2[i+1],y_2_eul2[i+1]}')
    if i == 30:
        print(f'At 30: {y_1_eul2[i+1],y_2_eul2[i+1]}')
    # print(y_1_eul[i],y_2_eul[i])

# ------------------------------------------------------
# plot results
plt.plot(x_eul, y_1_eul2 , 'b.-',x_eul, y_2_eul2 , 'r-')
plt.plot(x_eul, y_1_eul , 'b.-',x_eul, y_2_eul , 'r-')
plt.xlabel('time')
plt.ylabel('y_1(x), y_2(x)')
plt.show()

#%% QUESTION 2 OPTION 2
from scipy.integrate import solve_ivp
# ------------------------------------------------------
# functions that returns dy/dx
# i.e. the equation we want to solve:
def model(x,y):
    #rho = 10.0
    y_1 = y[0]
    y_2 = y[1]
    f_1 = y_1
    f_2 = -y_2*math.sin(x)
    return [f_1 , f_2]
# ------------------------------------------------------
# initial conditions
omega = 0
g = 9.81
l = 9.81
# ------------------------------------------------------
# initial conditions
x0 = 3.14159/4
y0_1 = omega
y0_2 = -g/l

# total solution interval
t_final = 30
# step size
# not needed here. The solver solve_ivp
# will take care of finding the appropriate step
# ------------------------------------------------------
# Apply solve_ivp method
t_eval = np.linspace(0, t_final, num=5000)
y = solve_ivp(model, [0 , t_final] ,
[y0_1 , y0_2],t_eval=t_eval)

plt.figure(1)
plt.plot(y.t,y.y[0,:] , 'b-',y.t,y.y[1,:], 'r-')
plt.xlabel('t')
plt.ylabel('y_1(t), y_2(t)')
# ------------------------------------------------------
# ------------------------------------------------------
# plot results
plt.figure(2)
plt.plot(y.y[0,:] ,y.y[1,:],'-')
plt.xlabel('y_1')
plt.ylabel('y_2')
# ------------------------------------------------------
# ------------------------------------------------------
# plot results
# ------------------------------------------------------
# ------------------------------------------------------
plt.show()

#%% QUESTION 2 OPTION 3
import numpy as np
from numpy import sin, cos
from scipy.integrate import odeint
from matplotlib import pyplot as plt
# define the equations
def equations(y0, t):
    theta, x = y0
    f = [x, -(g/l) * sin(theta)]
    return f
def plot_results(time, theta1, theta2):
    plt.plot(time, theta1[:,0])
    plt.plot(time, theta2)
    s = '(Initial Angle = ' + str(initial_angle) + ' degrees)'
    plt.title('Pendulum Motion: ' + s)
    plt.xlabel('time (s)')
    plt.ylabel('angle (rad)')
    plt.grid(True)
    plt.legend(['nonlinear', 'linear'], loc='lower right')
    plt.show()
# parameters
g = 9.81
l = 9.81
time = np.arange(0, 30, 0.025)
# initial conditions
initial_angle = 45
theta0 = np.radians(initial_angle)
x0 = np.radians(0.0)
# find the solution to the nonlinear problem
theta1 = odeint(equations, [theta0, x0], time)
# find the solution to the linear problem
w = np.sqrt(g/l)
theta2 = [theta0 * cos(w*t) for t in time]
# plot the results
plot_results(time, theta1, theta2)

#%% QUESTION 1
TA = 90
x = 50
y = 15
y0 = 5
w = 10

y = sp.Function('y')
x = sp.Symbol('x')
diff_equation = sp.Function('diff_equation')

y = TA/w*1/2*(2.7159**(w/TA*x)+2.7159**(-w/TA*x))+y0-TA/w

dy = sp.diff(y)
ddy = sp.diff(dy)

print(f'derivative {dy}')
print(f'derivative2 {ddy}')

diff_equation = w/TA*(1+( 0.499561695083684*2.7159**(0.111111111111111*x) - 0.499561695083684*2.7159**(-0.111111111111111*x))**2)**(1/2)

x_values = np.linspace(0,100,100)
ddy_values = []
b = ddy.evalf(subs = {x:10})
g = diff_equation.evalf(subs = {x:10})
diff_equation_values = []
for i in x_values:
    ddy_values.append(diff_equation.evalf(subs = {x:i}))
    diff_equation_values.append(diff_equation.evalf(subs = {x:i}))


plt.plot(x_values,ddy_values, label = 'graph 1')
plt.plot(x_values,diff_equation_values, label = 'graph 2')
plt.legend()
plt.show()

print(f'BY INDUCTIVE REASONING close enough: (difference) {g-b}')

"""IN ORDER TO PROVE THAT THE DIFFERENTIAL EQUATION IS EQUAL TO THE EQUATION, I TOOK THE SECOND DERIVATIVE OF THE EQUATION
AND CHECKED WHETHER IT EQUALS THE SAME AS THE DIFFERENTIAL EQUATION AFTER PLUGGING THE DERIVATIVE OF THE EQUATION INTO THE
DIFFERENTIAL EQUATION. FOR A RANDOM VALUE IT WAS CLOSE ENOUGH WITH A DIFFERENCE OF 0.02% BY INDUCTIVE REASONING ONE COULD 
SAY THAT IT IS THE SAME IF WE CHECKED FOR A RANDOM VALUE, HOWEVER IN ORDER TO FULLY VISUALLY PROVE THAT IT WORKS, I PLOTTED IT"""
























