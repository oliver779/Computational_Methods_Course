"""Solve the Diophantine Equation ax + by = c"""
a = 9
b = 3
c = 7
def solution(a,b,c):
    i = 0
    while i*a<=c:
        if (c - a*i) % b == 0:
            return  print(f'x = {i}  y = {int((c - i*a)/b)}')
        i += 1
    else:
        return print('no solution')
        
print(solution(a,b,c))


"""Extended Euclid GCD"""

def extended_euclid_gcd(a,b):
    s, old_s = 0,1
    t, old_t = 1,0
    r, old_r = b,a
    while r != 0:
        quotient = old_r/r
        old_r, r = r, old_r-quotient*r
        old_s, s = s, old_s-quotient*s
        old_t, t = t, old_t-quotient*t
    return [old_r, old_s, old_t]

res = extended_euclid_gcd(a, b) #c by definition is the GCD of these a and b.

print ('GCD of a and b is %d. x = %d and y = %d in ax + by = gcd(a, b)' % (res[0], res[1], res[2]))