""" Find the sum of the squares of the first 20 natural numbers"""
def square_sum():
    sm = 0
    i = 0
    n = 20
    while i<=n-1:
        sm+=i**2
        i+=1
    return sm
        
print(square_sum())
