# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 22:24:26 2021

@author: Oliver
"""
#Exercise 2: Use the pseudocode to factorise the following quadratic equation by (x-4):
#pseudo code
# lc;
# clear;
# n=2;
# a=zeros(n+1,1);
# a(1,1)=-24;
# a(2,1)=2;
# a(3,1)=1;
# t=-6;
# r=a(n+1,1);
# a(n+1,1)=0;
# for i = n:-1:1;
#     s=a(i,1);
#     a(i,1)=r;
#     r=s+r*t
# end
# my code
# a =[1,2,-24]
# r = 4
# t = 4
# n = 2
# for i in range(0, n-1):
#     s = a[i]
#     a[i] = r
#     r = s+r*t
# print(r)

# my code
p = [1,2,-24]
guess = 4
values =[p[0]]
z = p[0]
for i in range(0,2):
    x = p[i+1] + z*4
    z = x
    values.append(x)
print(values)

# n=2;
# a=zeros(n+1,1)
# a = [1,2,-24]
# t=-6
# #r=a(n+1,1)
# #a(n+1,1)=0
# for i in range(-1,1):
#     s=a[i]
#     a[i]=r
#     r=s+r*t







