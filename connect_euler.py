import time
start = time.time()
m = 55.5 #kilograms
c = 12 # drag coefficient in kilograms/second
g = 9.81 # acceleration of earth
t = 0 # time in question that we need to find the velocity at
step_size = 2 # time between two slopes
v_init = 12.4 # initial velocity

while t<=6: 
    t_new = t+step_size
    slope = (g-c/m*v_init)*(t_new-t)
    v_new = v_init + slope
    print(v_new)
    t=t_new
    v_init = v_new
    print(t)
print(f"The slope is: {slope} and the new velocity is: {v_new}")
    
print(f"The velocity at time of {t_new} seconds was {v_new}")
end = time.time()
print(f"The time it took to do this calculation was {end-start}")