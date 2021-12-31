"""Bracketing Methods"""
import math as m

"""Bisection Method with error""" """ROOT FINDING"""

def f(x):
    return x**3 - 4*(x*2) + 10

x_lower = 200
x_upper = -300
counter = 0
error = 0.000001
while True:
    counter +=1
    x_mid = (x_lower+x_upper)/2
    if f(x_lower)*f(x_upper)>0:
        print('The upper and lower limits have not been proper')
        break
    if f(x_lower)*f(x_mid)<0:
        x_upper = x_mid
    if f(x_lower)*f(x_mid)>0:
        x_lower = x_mid
    if -error <= f(x_lower)*f(x_mid) <= error:
        print(f'We have found the root at x = {x_mid} after {counter} iterations')
        break
    
    
"""False Position Method""" """ROOT FINDING""" """Similar Triangle method for finding the roots"""
MAX_ITER = 1000000 
a = 200
b = -300

# The function is x^3 - x^2 + 2 
def func(x): 
    return (x**3 - 4*(x*2) + 10) 
# Prints root of func(x) in interval [a, b] 
def regulaFalsi(a,b): 
    if func(a) * func(b) >= 0: 
        print("You have not assumed right a and b") 
        return -1
    c = a 
    # Initialize result 
    for i in range(MAX_ITER): 
        # Find the point that touches x axis 
        c = (a * func(b) - b * func(a))/ (func(b) - func(a)) 
        # Check if the above found point is root 
        if func(c) == 0: 
            break 
        # Decide the side to repeat the steps 
        elif func(c) * func(a) < 0: 
            b = c 
        else: 
            a = c 
    print("The value of root is : " , '%.4f' %c) 
    return
regulaFalsi(a,b)

"""Modified false Position Method""" """ROOT FINDING"""

def modregulaFalsi(a,b): 
    if func(a) * func(b) >= 0: 
        print("You have not assumed right a and b") 
        return -1
    c = a 
   # Initialize result 
    for i in range(MAX_ITER): 
    # Find the point that touches x axis 
        c1 = (a * 0.5*func(b) - b * func(a))/(0.5*func(b) - func(a))
        c2 = (a * func(b) - b * 0.5* func(a))/(func(b) - 0.5* func(a))
        if func(c1) < func(c2): 
            c = c1
        else: 
            c = c2
        # Check if the above found point is root 
        if func(c) == 0: 
            break 
        elif func(c) * func(a) < 0: 
            b = c 
        else: 
            a = c 
            print("The value of root is : " , '%.4f' %c) 
            break
        
modregulaFalsi(a,b)

"""Incremental Search Techniques for finding loads of roots""" """ROOT FINDING"""

def naive_root(f, x_guess, tolerance, step_size):
    steps_taken = 0
    while abs(f(x_guess)) > tolerance:
        if f(x_guess) > 0:
            x_guess -= step_size
        elif f(x_guess) < 0:
            x_guess += step_size
        else:
            return x_guess
        steps_taken += 1
    return x_guess, steps_taken

f = lambda x: x**2 - 20
root, steps = naive_root(f, x_guess=4.5, tolerance=.01, step_size=.001)
print ("root is:", root)
print ("steps taken:", steps)

"""Open Root Finding Methods""" """ROOT FINDING"""

def f(x):
    return x*x*x + x*x -1
# Re-writing f(x)=0 to x = g(x)
def g(x):
    return 1/m.sqrt(1+x)
# Implementing Fixed Point Iteration Method
def fixedPointIteration(x0, e, N): 
    print('\n\n*** FIXED POINT ITERATION ***')
    step = 1
    flag = 1
    condition = True
    while condition:
        x1 = g(x0)
        print('Iteration-%d, x1 = %0.6f and f(x1) = %0.6f' % (step, x1, f(x1)))
        x0 = x1
        step = step + 1 
        if step > N:
            flag=0
            break 
        condition = abs(f(x1)) > e
        if flag==1:
            print('\nRequired root is: %0.8f' % x1)
        else:
            print('\nNot Convergent.')
            
# Input Section
x0 = 1
e = 0.01
N = 1000

fixedPointIteration(x0, e, N)    
        
"""Newton Raphson""" """ROOT FINDING""" """BEST TO USE BUT HAS PROBLEMS WITH DERIVATIVE IF EQUATION IS DIFFICULT"""

def newton(f,Df,x0,epsilon,max_iter):
    """Solution of f(x)=0 by Newton's method.
    Parameters
    ----------
    f : function for which we are searching for a solution 
    f(x)=0.
    Df : Derivative of f(x).
    x0 : Initial guess for a solution f(x)=0.
    epsilon : number
    Stopping criteria is abs(f(x)) < epsilon.
    max_iter : integer
    Maximum number of iterations
    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> Df = lambda x: 2*x - 1
    >>> newton(f,Df,1,1e-8,10)
    Found solution after 5 iterations.
    1.618033988749989  """       
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

f = lambda x: x**4 - x - 1
df= lambda x: 4*x**3 - 1
x0=1
epsilon=0.001
max_iter=100
solution = newton(f,df,x0,epsilon,max_iter)
print(f'solution {solution}')
      
"""Secant Technique"""

def secant(f,a,b,N): 
    if f(a)*f(b) >= 0: 
        print("Secant method fails.") 
        return None
    a_n = a
    b_n = b 
    for n in range(1,N+1): 
        m_n = a_n - f(a_n)*(b_n -a_n)/(f(b_n) - f(a_n)) 
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
            
f = lambda x: x**4 - x - 1
solution = secant(f,1,2,25)
print(f'Secant Method solutions: {solution}')

"""Modified Secant Technique"""        
        
def secant_method(func, x0, alpha=1.0, tol=1E-9, maxit=200):
    """
    Uses the secant method to find f(x)=0. 
    INPUTS
    * f : function f(x)
    * x0 : initial guess for x
    * alpha : relaxation coefficient: 
    modifies Secant step size
    * tol : convergence tolerance
    * maxit : maximum number of iteration, default=200 
    """
    x, xprev = x0, 1.001*x0
    f, fprev = x**4 - x - 1, xprev**4 - xprev - 1
    rel_step = 2.0 *tol
    k = 0
    rel_step = abs(x-xprev)/abs(x)
    # Full secant step
    dx = -f/(f - fprev)*(x - xprev)
    while (abs(f) > tol) and (rel_step) > tol and (k<maxit): 
        if x == 0:
            x = 1
        rel_step = abs(x-xprev)/abs(x)
    # Full secant step
        dx = -f/(f - fprev)*(x - xprev)
    # Update `xprev` and `x` simultaneously
        xprev, x = x, x + alpha*dx
    # Update `fprev` and `f`:
        fprev, f = f, func(x)
        k += 1
        # print('{0:10d} {1:12.5f} {2:12.5f}{3:12.5f} {4:12.5f}'\.format(k, xprev, fprev, rel_step, alpha*dx))
    return x
funcT = lambda x: x**4 - x - 1
solution = secant_method(funcT,1,alpha=1.0,tol=1E-9,maxit=200)
print(f'Modified Secant Method solutions: {solution}')


"""Inverse Quadratic Interpolation"""

def inverse_quadratic_interpolation(f, x0, x1, x2, max_iter=20, 
    tolerance=1e-5):
    steps_taken = 0
    while steps_taken < max_iter and abs(x1-x0) > tolerance: # last guess and new guess are v close
        fx0 = f(x0)
        fx1 = f(x1)
        fx2 = f(x2)
        L0 = (x0 * fx1 * fx2) / ((fx0 - fx1) * (fx0 - fx2))
        L1 = (x1 * fx0 * fx2) / ((fx1 - fx0) * (fx1 - fx2))
        L2 = (x2 * fx1 * fx0) / ((fx2 - fx0) * (fx2 - fx1))
        new = L0 + L1 + L2
        x0, x1, x2 = new, x0, x1
        steps_taken += 1
    return x0, steps_taken
f = lambda x: x**2 - 20
root, steps = inverse_quadratic_interpolation(f, 4.3, 4.4, 4.5)
print ("Inverse Quadratic Interpolation root is:", root)
print ("steps taken:", steps)

"""Brent Method"""
def f_01 ( x ):
  import numpy as np
  value = np.sin ( x ) - 0.5 * x
  return value


def zero ( a, b, machep, t, f ):

#*****************************************************************************80
#
## ZERO seeks the root of a function F(X) in an interval [A,B].
#
#  Discussion:
#
#    The interval [A,B] must be a change of sign interval for F.
#    That is, F(A) and F(B) must be of opposite signs.  Then
#    assuming that F is continuous implies the existence of at least
#    one value C between A and B for which F(C) = 0.
#
#    The location of the zero is determined to within an accuracy
#    of 6 * MACHEPS * abs ( C ) + 2 * T.
#
#    Thanks to Thomas Secretin for pointing out a transcription error in the
#    setting of the value of P, 11 February 2013.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    03 December 2016
#
#  Author:
#
#    Original FORTRAN77 version by Richard Brent
#    Python version by John Burkardt
#
#  Reference:
#
#    Richard Brent,
#    Algorithms for Minimization Without Derivatives,
#    Dover, 2002,
#    ISBN: 0-486-41998-3,
#    LC: QA402.5.B74.
#
#  Parameters:
#
#    Input, real A, B, the endpoints of the change of sign interval.
#
#    Input, real MACHEP, an estimate for the relative machine
#    precision.
#
#    Input, real T, a positive error tolerance.
#
#    Input, real value = F ( x ), the name of a user-supplied
#    function which evaluates the function whose zero is being sought.
#
#    Output, real VALUE, the estimated value of a zero of
#    the function F.
#

#
#  Make local copies of A and B.
#
  sa = a
  sb = b
  fa = f ( sa )
  fb = f ( sb )

  c = sa
  fc = fa
  e = sb - sa
  d = e

  while ( True ):
    if ( abs ( fc ) < abs ( fb ) ):
      sa = sb
      sb = c
      c = sa
      fa = fb
      fb = fc
      fc = fa
    tol = 2.0 * machep * abs ( sb ) + t
    m = 0.5 * ( c - sb )
    if ( abs ( m ) <= tol or fb == 0.0 ):
      break
    if ( abs ( e ) < tol or abs ( fa ) <= abs ( fb ) ):
      e = m
      d = e
    else:
      s = fb / fa
      if ( sa == c ):
        p = 2.0 * m * s
        q = 1.0 - s
      else:
        q = fa / fc
        r = fb / fc
        p = s * ( 2.0 * m * q * ( q - r ) - ( sb - sa ) * ( r - 1.0 ) )
        q = ( q - 1.0 ) * ( r - 1.0 ) * ( s - 1.0 )
      if ( 0.0 < p ):
        q = - q
      else:
        p = - p
      s = e
      e = d
      if ( 2.0 * p < 3.0 * m * q - abs ( tol * q ) and p < abs ( 0.5 * s * q ) ):
        d = p / q
      else:
        e = m
        d = e
    sa = sb
    fa = fb
    if ( tol < abs ( d ) ):
      sb = sb + d
    elif ( 0.0 < m ):
      sb = sb + tol
    else:
      sb = sb - tol
    fb = f ( sb )
    if ( ( 0.0 < fb and 0.0 < fc ) or ( fb <= 0.0 and fc <= 0.0 ) ):
      c = sa
      fc = fa
      e = sb - sa
      d = e
  value = sb
  return value

def zero_test ( ):

#*****************************************************************************80
#
## ZERO_TEST tests the Brent zero finding routine on all test functions.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    03 December 2016
#
#  Author:
#
#    John Burkardt
#
  import numpy as np

  print ( '' )
  print ( 'ZERO_TEST' )
  print ( '  ZERO seeks a root X of a function F()' )
  print ( '  in an interval [A,B].' )

  eps = 2.220446049250313E-016
  machep = np.sqrt ( eps )
  t = 10.0 * np.sqrt ( eps )
  a = 1.0
  b = 2.0
  x = zero ( a, b, machep, t, f_01 )
  print ( '' )
  print ( '  f_01(x) = sin ( x ) - x / 2' )
  print ( '  f_01(%g) = %g' % ( x, f_01 ( x ) ) )

def timestamp ( ):

#*****************************************************************************80
#
## TIMESTAMP prints the date as a timestamp.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license. 
#
#  Modified:
#
#    06 April 2013
#
#  Author:
#
#    John Burkardt
#
#  Parameters:
#
#    None
#
  import time

  t = time.time ( )
  print ( time.ctime ( t ) )

  return None

def timestamp_test ( ):

#*****************************************************************************80

  import platform

  print ( '' )
  print ( 'TIMESTAMP_TEST:' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  TIMESTAMP prints a timestamp of the current date and time.' )
  print ( '' )

  timestamp ( )
#
#  Terminate.
#
  print ( '' )
  print ( 'TIMESTAMP_TEST:' )
  print ( '  Normal end of execution.' )
  return

if ( __name__ == '__main__' ):
  timestamp ( )
  zero_test ( )
  timestamp ( )
  
"""Ralston - Rabinowitz for Multiple Roots"""

function = lambda x: (x-3)*(x-1)*(x-1)
derivative = lambda x: 3*x**2-10*x+7
second_derivative = lambda x: 6*x -10

def u(x):
    return function(x)/derivative(x)

def derivative_u(x):
    return (derivative(x)**2-function(x)*second_derivative(x))/(derivative(x)**2)


def Rolston_Rabinowitz(f,Df,x0,epsilon,max_iter):
    """Solution of f(x)=0 by Newton's method.
    Parameters
    ----------
    f : function for which we are searching for a solution 
    f(x)=0.
    Df : Derivative of f(x).
    x0 : Initial guess for a solution f(x)=0.
    epsilon : number
    Stopping criteria is abs(f(x)) < epsilon.
    max_iter : integer
    Maximum number of iterations
    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> Df = lambda x: 2*x - 1
    >>> newton(f,Df,1,1e-8,10)
    Found solution after 5 iterations.
    1.618033988749989  """       
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
    xn = xn - u(xn)/(derivative_u(xn))
    print('Exceeded maximum iterations. No solution found.')
    return None


x0=1
epsilon=0.01
max_iter=100
solution = newton(function,derivative,x0,epsilon,max_iter)
print(f'The solution of Rolstion using derivatives is: {solution}')


















