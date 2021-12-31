# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 10:29:13 2021

@author: Oscar
"""

"""
# Calulate the sum of squares of however many natural numbers
def squaresum(n):
    #Initiate a variable for holding the sums
    sm = 0
    # Iterate the addition of each individual squares from 1 to n+1 number
    for i in range(1, n+1):
        sm = sm + (i*i) # Adding each iteration of , for example 2x2 to 1x1 etc
    return sm #returns the value of the summation

#Drivers for the program
n = 100
print(squaresum(n))
"""
# Calculate the sum of the squares of the first 20 odd natural numbers
    #Natural number: Numbers used" for counti"ng (1,2,3,4",5,...)
# User input command
n = int(input("Print sum of square of first odd numbres up to the following number:")) # Input must be an integer, hence "int"

def squaresum(n):
    #Initiate a variable for holding the sums
    sm = 0
    # Iterate the addition of each individual squares from 1 to n+1 number
    for i in range(1, n+1):
        # determine in i in range(1, n+1) is odd
        if (i % 2 != 0): # Modulo Operator "%" finds the remainder of the specified values
            sm = sm + (i*i) # Adding each iteration of , for example 2x2 to 1x1 etc
    return sm #returns the value of the summation

#Drivers for the program
print("Sum of Squares of odd numbers from 1 to", n, "is :",squaresum(n))
