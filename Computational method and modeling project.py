


# Form a uniform and random distribution of particles within that grid - DONE
# Need someone to solve the SDEs and append new values in a efficient and innovative manner for each random uniform point - DONE
# Use Euler MAruyama to approximate the next values of the particles - DONE
# point extraction and spatial interpolation in python -DONE
# Need someone to interpolate the velocity function onto the grid - DONE
# make the edges of the grid bouncy walls so that particles bounce from it, bouncing between particles is non-existent
# Find the error involved from that and compare it to the true value
# density field 
# colors for milk and coffee and the dot of milk in coffee there must be a way to change the color of the particle 
# if it is in a certain region something that does a circle and then you import values into that region in a different 
# color
# make a file for the inputs of the simulation
# First start with intial values as being zero, then we will combine it with the dat file
# Need someone to do the UI for the app that we will build
# solve the problem
# Need to write the report(once we finish all of the above) 5 pages

import timeit
import functions_project

#Variables
 
#D = float(input("Please give the number for Diffusivity 'D': ")) #Diffusivity
#h = 1 #timestep
#s_t = 1 #solution time
#N_x = 1 #Number of grid-points in x-direction
#N_y = 1 #Number of grid-points in y-direction
#x_min = 1 #size of domain minimum x-value
#x_max = 1 #size of domain maximum x-value
#y_min = 1 #size of domain minimum y-value
#y_max = 1 #size of domain maximum y-value
#v = 1 #velocity (zero or imported from file)
#Np = 65000 #Number of particles

n = 100
t=0

functions_project.random_initial_values(n,functions_project.xllim,functions_project.xulim,functions_project.yllim,functions_project.yulim)

while t<0.2:
    start1 = timeit.default_timer()
    functions_project.plot(n,functions_project.xllim,functions_project.xulim,functions_project.yllim,functions_project.yulim)
    functions_project.counter(n,functions_project.x,functions_project.y)
    t+=functions_project.h
    stop1 = timeit.default_timer()
    print('Time of changed: ', stop1 - start1)  
    



# Equations to be implemented
#functions_project.plot(n,xllim,xulim,yllim,yulim)

# for count in range(0, 100):
#     for element, element1 in zip(functions_project.x, functions_project.y):
#         start = timeit.default_timer()
#         functions_project.Euler_Maruyama_x_position(n, variance, mean, element, element)
#         stop = timeit.default_timer()
#         print('Time of euler x position: ', stop - start)
#         start1 = timeit.default_timer()
#         functions_project.Euler_Maruyama_y_position(n, variance, mean, element1, element1)
#         stop1 = timeit.default_timer()
#         print('Time of euler y position: ', stop1 - start1)
#         start2 = timeit.default_timer()
#         functions_project.x = functions_project.x_new 
#         stop2 = timeit.default_timer()
#         print('appending x values: ', stop2 - start2)
#         start3 = timeit.default_timer()
#         functions_project.y = functions_project.y_new 
#         stop3 = timeit.default_timer()
#         print('appending y values: ', stop3 - start3)
#     functions_project.plot(n,xllim,xulim,yllim,yulim)
#     count +=1
#     print(count)































































