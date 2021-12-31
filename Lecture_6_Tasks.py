"""Solving the Colebrook Equation Using Different Numerical Methods"""

import math
import numpy as np
import matplotlib.pyplot as plt
import timeit


# Here are all the variables that will be
# Needed for the calculations
velocity = 40
density = 1.23
viscosity = 0.0000179
diameter = 0.005
roughness =0.0015
xlist= np.linspace(0.03,0.05,10000)
R =13743.016759776536


def Colebrook(x,roughness,diameter,R):
    return 1/math.sqrt(x)+ 2*math.log((roughness)/(3.7*diameter)+2.51/(R*math.sqrt(x)))

ylist = []
for element in xlist:
    ylist.append(Colebrook(element,roughness,diameter,R))

plt.figure(dpi=150)
plt.plot(xlist,ylist,'r')
plt.axhline(0, color='green')
plt.show

def bisection(f,a,b,N):
    if f(a)*f(b) >= 0:
        print("Bisection method fails.")
        return None
    a_n = a
    b_n = b
    for n in range(1,N+1):
        m_n = (a_n + b_n)/2
        f_m_n = f(m_n)
        if f(a_n)*f_m_n < 0:
                a_n = a_n
                b_n = m_n
        elif f(b_n)*f_m_n < 0:
                a_n = m_n
                b_n = b_n
        elif f_m_n == 0:
            print(f"Found exact solution after {n} tries")
            return m_n
        else:
            print("Bisection method fails.")
            return None
    return (a_n + b_n)/2
a= 0.03
b=0.045
n=50
fa = lambda x :1/math.sqrt(x)+ 2*math.log((roughness)/(3.7*diameter)+2.51/(R*math.sqrt(x)))
startb = timeit.default_timer()
approx_phi = bisection(fa,a,b,n)
stopb = timeit.default_timer()
print('Time of Bisection method: ', stopb - startb)
print(f'Using the bisection technique w/o error we get {approx_phi} after {n} tries')

def f(x):
    f = 1/math.sqrt(x)+ 2*math.log((roughness)/(3.7*diameter)+2.51/(R*math.sqrt(x)))
    return f

def regulaFalsi2(a,b,TOL,N):
    i = 1
    FA = f(a)
    print("%-20s %-20s %-20s %-20s %-20s" % ("n","a_n","b_n","p_n","f(p_n)"))
    while(i <= N):
            p = (a*f(b)-b*f(a))/(f(b) - f(a))
            FP = f(p)
            if(FP == 0 or np.abs(f(p)) < TOL):
                break
            else:
                print("%-20.8g %-20.8g %-20.8g %-20.8g %-20.8g\n" % (i, a, b, p, f(p)))
            i = i + 1
            if(FA*FP > 0):
                a = p
            else:
                b = p
    return 
    print(i)

startf = timeit.default_timer()
regulaFalsi2(a, b, 0.0001, 100)
stopf = timeit.default_timer()
print('Time of False position method: ', stopf - startf)

def newton(f,df,x,epsilon,max_iter):
    xn = x
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

Df= lambda x: (-0.00112627 - 0.502253*math.sqrt(x))/(0.00225254*x**(1.5)+x**2)
epsilon=0.000000000000001
max_iter=100
startn = timeit.default_timer()
solution = newton(fa,Df,a,epsilon,max_iter)
stopn = timeit.default_timer()
print('Time of original newton: ', stopn - startn) 
print(solution)



def secant(f,a,b,N): 
    if f(a)*f(b) >= 0: 
        print("Secant method fails.") 
        return None 
    a_n = a
    b_n = b 
    for n in range(1,N+1): 
        m_n = a_n - f(a_n)*(b_n - a_n)/(f(b_n) - f(a_n)) 
        f_m_n = f(m_n) 
        if f(a_n)*f_m_n < 0: 
            a_n = a_n 
            b_n = m_n
        elif f(b_n)*f_m_n < 0: 
            a_n = m_n 
            b_n = b_n
        elif f_m_n == 0: 
            print("Found exact solution.") 
            return m_n
        else:
            print("Secant method fails.") 
            return None 
    return a_n - f(a_n)*(b_n - a_n)/(f(b_n) - f(a_n))

n2 = 100
starts = timeit.default_timer()
solution = secant(fa,a,b,n2)
stops = timeit.default_timer()
print('Time of original secant: ', stops - starts) 
print(solution)

