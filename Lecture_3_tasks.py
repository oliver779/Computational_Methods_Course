""" Approximate e**0.5 Numerically and add error analysis"""

import math as m
n = 0.5
accuracy = 10000000
error = 0.0001
Analytical_value = m.exp(n)
print(f'The Analytical solution is {Analytical_value}')

def numerical_e(n):
    sm = 0
    i = 0
    accuracy = 6
    while i <= accuracy:
        diff = (Analytical_value - sm)/Analytical_value
        if i == 0:
            sm += 1
        i += 1
        sm += n**i/m.factorial(i)
        if diff <= error:
            print(f'The numerical solution is within error and the value is {sm}')
            print(f'The relative error between the analytical and computational is {diff*100}%')
            break
    return sm

numerical_e(n)