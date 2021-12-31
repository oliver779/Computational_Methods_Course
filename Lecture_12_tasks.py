# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 19:35:06 2021

@author: Oliver
"""

"""Linear Algebra Numerical Solving Methods"""

"""Gauss Elimination"""

import numpy as np

def linearsolver(A,b):
    n = len(A)
    #Initialise solution vector as an empty array
    x = np.zeros(n)
    #Join A and use concatenate to form an augmented coefficient matrix
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
    #Python starts indexing with 0, so the last element is n-1
    x[n-1] =M[n-1][n]/M[n-1][n-1]
    #We need to start at n-2, because of Python indexing
    for i in range (n-2,-1,-1):
        z = M[i][n]
        for j in range(i+1,n):
            z = z - M[i][j]*x[j]
        x[i] = z/M[i][i]
    return x

#Initialise the matrices to be solved.
A=np.array([[10., 15., 25],[4., 5., 6],[25, 3, 8]])
b=np.array([[34., 25., 15]])
print(f'Gauss Elimination: {linearsolver(A,b)}')


"""Gaussian Elimination with Partial Pivoting"""
def column(m, c):
    return [m[i][c] for i in range(len(m))]

def row(m, r):
    return m[r][:]

def height(m):
    return len(m)

def width(m):
    return len(m[0])

def print_matrix(m):
    for i in range(len(m)):
        print(m[i])
        
def gaussian_elimination_with_pivot(m):
# forward elimination
    n = height(m)
    for i in range(n):
        pivot(m, n, i)
        for j in range(i+1, n):
            m[j] = [m[j][k] - m[i][k]*m[j][i]/m[i][i] for k in range(n+1)]
    if (m[n-1][n-1]) == 0:
        raise ValueError('No unique solution')
    # backward substitution
    x = [0] * n
    for i in range(n-1, -1, -1):
        s = sum(m[i][j] * x[j] for j in 
        range(i, n))
        x[i] = (m[i][n] - s) / m[i][i]
    return x
    
def pivot(m, n, i):
    max = -1e100
    for r in range(i, n):
        if max < abs(m[r][i]):
            max_row = r
            max = abs(m[r][i])
        m[i], m[max_row] = m[max_row], m[i]
        
if __name__ == '__main__':
    #m = [[0,-2,6,-10], [-1,3,-6,5], [4,-12,8,12]]
    #m = [[1,-1,3,2], [3,-3,1,-1], [1,1,0,3]]
    m = [[1,-1,3,2], [6,-6,2,-2], [1,1,0,3]]
    print(f'Guassian pivot:{gaussian_elimination_with_pivot(m)}')
    
    





















