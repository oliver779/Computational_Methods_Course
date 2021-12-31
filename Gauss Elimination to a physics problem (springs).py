# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 21:11:58 2021

@author: Oscar
"""
import numpy as np

def linearsolver(A,b):
    n = len(A)
    
    #Initialise solution vector as an empty array
    x = np.zeros(n)
    
    # Join A and use concatenate to form an 
    # augmented coefficient matrix
    M = np.concatenate((A,b.T), axis=1)
    
    for k in range(n):
        for i in range(k,n):
            if abs(M[i][k]) > abs(M[k][k]):
                M[[k,i]] = M[[i,k]]
            else:
                pass
                for j in range(k+1,n):
                    q = M[j][k] / M[k][k]
                    for m in range(n+1):
                        M[j][m] += -q * M[k][m]
    #Python starts indexing with 0, so the last 
    #element is n-1
    x[n-1] =M[n-1][n]/M[n-1][n-1]
    
    #We need to start at n-2, because of
    #Python indexing
    for i in range (n-2,-1,-1):
        z = M[i][n]
        for j in range(i+1,n):
            z = z - M[i][j]*x[j]
        x[i] = z/M[i][i]
    return x

#Initialise the matrices to be solved.
A = np.array([[30., -20., 0],[-20., 30., -10.],[0, -10, 10]])
b = np.array([[19.62, 29.43, 24.525]])
print(linearsolver(A,b))