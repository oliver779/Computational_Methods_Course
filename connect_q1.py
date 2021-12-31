import math
import time
start = time.time()
m = 67.5 #kilograms
c = 13.45 # drag coefficient in kilograms/second
g = 9.81 # acceleration of earth
t = 7 # time in question that we need to find the velocity at
def v(t):
    return g*m/c*(1-math.exp(-(c/m)*t))
print(f"The velocity at time of {t} seconds was {v(t)}")
end = time.time()
print(f"The time it took to do this calculation was {end-start}")