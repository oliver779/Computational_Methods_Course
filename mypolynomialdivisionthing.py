# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 20:09:27 2021

@author: Oliver
"""
import timeit
n=0
epsilon = 0.5
p =[4,2,0,0]
x=int(input("Please give a value of x as a guess (integer): "))
def repeating_test(x,n,e):
    f = p[-1]
    n = n+1
    f2 = p[-1]
    z =len(p)-1
    for element in p:
        x_new = x + 1
        f = f + element*x**z
        f2 = f2 + element*x_new**z
        z = z-1
        if z==0:
            if -e<=f2<=e:
                print(f"We have found it, after {n} tries the root of the polynomial is: {x_new}")
                break
            if -e<=f<=e:
                print(f"We have found it, after {n} tries the root of the polynomial is: {x}")
                break
            if f2 <= f:
                x = x + 2
                repeating_test(x,n,e)
            if f2 >= f:
                x = x - 2
                repeating_test(x,n,e)
        if z == 0 and f == 0:
            print(f"we have found a root with x equal to: {x}")
            break
    return   

starts = timeit.default_timer()
repeating_test(x,n,epsilon)
stops = timeit.default_timer()
print('Time of repeating test: ', stops - starts) 
