# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 17:53:01 2021

@author: Oliver
"""
A = 0.521
C = 0.665
D = 0.122
t=0
error = 0.00001
while t<1:
    g = (A/6.54 + C**2)/(t+D)
    if t-error<=g<=t+error:
        print(f"HOORAY my f is: {g}, for t: {t}")
        break
    t+=0.000001
    