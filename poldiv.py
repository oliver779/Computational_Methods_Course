# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 22:25:23 2021

@author: Oliver
"""

#Exercise 3:Apply this to the division of the 5th order polynomial above by the quadratic equation (x+1)(x-4)
#pseudo code is on the 15th slide
p = [1,2,3,4,5]
n = len(p-1)

def poldiv(a,n,d,m,q,r):
    for i in range(n):
        r(i)=a(i)
        q(i) =0
    if k =n-m or k =0 or k=-1:
        
