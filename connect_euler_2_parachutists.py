import math
import time
start = time.time()
m1 = 60 #kilograms
m2 = 80 #kilograms
c1 = 10 # drag coefficient in kilograms/second
c2 = 15 # drag coefficient in kilograms/second
g = 9.81 # acceleration of earth
t1 = 9 # time in question that we need to find the velocity at
t2 = 11 # time in question that we need to find the velocity at
step_size = 0.001 # time between two slopes
v_init = 12.4 # initial velocity

def v(t,m,c):
    return g*m/c*(1-math.exp(-(c/m)*t))
print(f"The velocity of the first parachutist at time of {t1} seconds was {v(t1,m1,c1)}")

while t2<=12:
    t_new = t2+step_size
    if "{:.4f}".format(v(t_new,m2,c2))=="{:.4f}".format(v(t1,m1,c1)):
        print(f"Eureka! We found the time at which both parachutists reach the same velocity. The second reaches the first's velocity after {t_new} seconds!")
        break
    else:
        continue
print(f"The velocity of the second parachutist at time of {t2} seconds was {v(t2,m2,c2)}")

# while t<=8: 
#     t_new = t+step_size
#     slope = (g-c1/m1*v_init)*(t_new-t)
#     v_new = v_init + slope
#     print(v_new)
#     t=t_new
#     v_init = v_new
#     print(t)
# print(f"The slope is: {slope} and the new velocity is: {v_new}")
    
# print(f"The velocity at time of {t_new} seconds was {v_new}")
end = time.time()
print(f"The time it took to do this calculation was {end-start}")