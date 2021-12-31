# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 12:22:38 2020

@author: emc1977
"""

import math as math

def function ( x ):
  import numpy as np
  #Note that here I have negatived all terms, converting the minimiser into a maximiser.
  #The functino value is returned as a negative, though in reality it is positive.
  value = -2*math.sin(x)+(x**2)/10
  return value

def glomin ( a, b, c, m, machep, e, t, f ):
  import numpy as np
  a0 = b
  x = a0
  a2 = a
  y0 = f ( b )
  yb = y0
  y2 = f ( a )
  y = y2
  if ( y0 < y ):
    y = y0
  else:
    x = a
  if ( m <= 0.0 or b <= a ):
    fx = y
    return x, fx
  m2 = 0.5 * ( 1.0 + 16.0 * machep ) * m
  if ( c <= a or b <= c ):
    sc = 0.5 * ( a + b )
  else:
    sc = c
  y1 = f ( sc )
  k = 3
  d0 = a2 - sc
  h = 9.0 / 11.0
  if ( y1 < y ):
    x = sc
    y = y1
  while ( True ):
    d1 = a2 - a0
    d2 = sc - a0
    z2 = b - a2
    z0 = y2 - y1
    z1 = y2 - y0
    r = d1 * d1 * z0 - d0 * d0 * z1
    p = r
    qs = 2.0 * ( d0 * z1 - d1 * z0 )
    q = qs
    if ( k < 1000000 or y2 <= y ):
      while ( True ):
        if ( q * ( r * ( yb - y2 ) + z2 * q * ( ( y2 - y ) + t ) ) < \
          z2 * m2 * r * ( z2 * q - r ) ):
          a3 = a2 + r / q
          y3 = f ( a3 )
          if ( y3 < y ):
            x = a3
            y = y3
        k = ( ( 1611 * k ) % 1048576 )
        q = 1.0
        r = ( b - a ) * 0.00001 * float ( k )
        if ( z2 <= r ):
          break
    else:
      k = ( ( 1611 * k ) % 1048576 )
      q = 1.0
      r = ( b - a ) * 0.00001 * float ( k )
      while ( r < z2 ):
        if ( q * ( r * ( yb - y2 ) + z2 * q * ( ( y2 - y ) + t ) ) < \
          z2 * m2 * r * ( z2 * q - r ) ):
          a3 = a2 + r / q
          y3 = f ( a3 )
          if ( y3 < y ):
            x = a3
            y = y3
        k = ( ( 1611 * k ) % 1048576 )
        q = 1.0
        r = ( b - a ) * 0.00001 * float ( k )
    r = m2 * d0 * d1 * d2
    s = np.sqrt ( ( ( y2 - y ) + t ) / m2 )
    h = 0.5 * ( 1.0 + h )
    p = h * ( p + 2.0 * r * s )
    q = q + 0.5 * qs
    r = - 0.5 * ( d0 + ( z0 + 2.01 * e ) / ( d0 * m2 ) )
    if ( r < s or d0 < 0.0 ):
      r = a2 + s
    else:
      r = a2 + r
    if ( 0.0 < p * q ):
      a3 = a2 + p / q
    else:
      a3 = r
    while ( True ):
      a3 = max ( a3, r )
      if ( b <= a3 ):
        a3 = b
        y3 = yb
      else:
        y3 = f ( a3 )
      if ( y3 < y ):
        x = a3
        y = y3
      d0 = a3 - a2
      if ( a3 <= r ):
        break
      p = 2.0 * ( y2 - y3 ) / ( m * d0 )
      if ( ( 1.0 + 9.0 * machep ) * d0 <= abs ( p ) ):
        break
      if ( 0.5 * m2 * ( d0 * d0 + p * p ) <= ( y2 - y ) + ( y3 - y ) + 2.0 * t ):
        break
      a3 = 0.5 * ( a2 + a3 )
      h = 0.9 * h
    if ( b <= a3 ):
      break
    a0 = sc
    sc = a2
    a2 = a3
    y0 = y1
    y1 = y2
    y2 = y3
  fx = y  
  print(x,fx)
  return x, fx
  
def glomin_test ( ):

  import numpy as np

  machep = 2.220446049250313E-016
  e = np.sqrt ( machep )
  t = np.sqrt ( machep )

  a = 0.1
  b = 2
  c = ( a + b ) / 2.0
  m = 3

  example_test ( a, b, c, m, machep, e, t, function, '.' )
#
#  Terminate.
#
  print ( '' )
  print ( 'GLOMIN_TEST' )
  print ( '  Normal end of execution.' )
  return

def example_test ( a, b, c, m, machep, e, t, f, title ):


  x, fx = glomin ( a, b, c, m, machep, e, t, f )
  fa = f ( a )
  fb = f ( b )

  print ( '' )
  print ( '  %s' % ( title ) )
  print ( '' )
  print ( '      A                 X             B' )
  print ( '    F(A)              F(X)          F(B)' )
  print ( '' )
  print ( '  %14f  %14f  %14f' % ( a,  x,  b ) )
  print ( '  %14e  %14e  %14e' % ( fa, fx, fb ) )

  return

def timestamp ( ):

  import time

  t = time.time ( )
  print ( time.ctime ( t ) )

  return None

def timestamp_test ( ):

  import platform

  print ( '' )
  print ( 'TIMESTAMP_TEST:' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  TIMESTAMP prints a timestamp of the current date and time.' )
  print ( '' )

  timestamp ( )

 # Terminate

  print ( '' )
  print ( 'TIMESTAMP_TEST:' )
  print ( '  Normal end of execution.' )
  return

if ( __name__ == '__main__' ):
  timestamp ( )
  glomin_test ( )
  timestamp ( )
