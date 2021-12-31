# In this tutorial you are given this:
n = 40
def squaresum(n) : 
# Iterate i from 1 
# to n finding 
# the square of i and 
# add to sum. 
    sm = 0
    for i in range(1, n+1) : 
        sm = sm + (i * i) 
    return sm
# Driven Program 

print(f"The sum of squares of all numbers is: {squaresum(n)}")
# You need to adapt this code to sum the squares of 
# the first n odd numbers only
def squaresum_odd(n) : 
    sm1 = 0
    for i in range(1, n+1) : 
        if (i % 2) == 0:
            continue
        else:
            sm1 = sm1 + (i * i) 
    return sm1


print("_______________________________________")
print(f"The sum of all od numbers is: {squaresum_odd(n)}")

def squaresum_odd_answer_book(n) : 
    sm2 = 0
    for i in range(1, n+1) : 
        if (not i % 2) == 0:
            sm2 = sm2 + (i * i) 
    return sm2

print("_______________________________________")
print(f"The sum of all odd numbers is (from answer): {squaresum_odd_answer_book(n)}")