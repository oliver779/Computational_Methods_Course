# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 15:10:34 2021

@author: Oliver
"""
import random

x = range(1, 10000)
y = range(1, 10000)
minAllowableDistance = 0.05
numberOfPoints = 300

% Initialize first point.
keeperX = x(1)
keeperY = y(1)
% Try dropping down more points.
counter = 2;
for k = 2 : numberOfPoints
  % Get a trial point.
  thisX = x(k);
  thisY = y(k);
  % See how far is is away from existing keeper points.
  distances = sqrt((thisX-keeperX).^2 + (thisY - keeperY).^2);
  minDistance = min(distances);
  if minDistance >= minAllowableDistance
    keeperX(counter) = thisX;
    keeperY(counter) = thisY;
    counter = counter + 1;
  end
end
plot(keeperX, keeperY, 'b*');
grid on;

int(input("Please give the number of points you would like to consider in the next simulation: "))

import math
import random
import matplotlib.pyplot as plt
# create random data
no_of_balls = 25
x = [random.triangular() for i in range(no_of_balls)]
y = [random.gauss(0.5, 0.25) for i in range(no_of_balls)]

# draw the plot
plt.figure()
plt.scatter(x, y)
plt.axis([0.0, 1.0, 0.0, 1.0])
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

#dXp = u(Xp,Yp)*dt+math.sqrt(2*D)*dWx
#dYp = v(Xp,Yp)*dt+math.sqrt(2*D)*dWy
nx_p = x_p+u(x_p,y_p)*h+math.sqrt(2*D*h*epsilon_x)
Yp_i+1 = Yp_i+v(Xp_i,Yp_i)*h+math.sqrt(2*D*h*epsilon_y)
This is the Euler Maruyama method
num_sims = 30 # Display five runs
Xp_i = 0
Yp_i = 0
u = 0
v = 0
t_init = 3
t_end  = 15
N      = 1000  # Compute 1000 grid points
h = t_end-t_init
dt     = float(t_end - t_init) / N
y_init = 0
c_theta = 0.7
c_mu    = 1.5
c_sigma = 0.06

def mu(y, t):
    """Implement the Ornstein–Uhlenbeck mu."""  # = \theta (\mu-Y_t)
    return c_theta * (c_mu - y)

def sigma(y, t):
    """Implement the Ornstein–Uhlenbeck sigma."""  # = \sigma
    return c_sigma

def dW(delta_t):
    """Sample a random number at each call."""
    return np.random.normal(loc=0.0, scale=np.sqrt(delta_t))

ts = np.arange(t_init, t_end + dt, dt)
ys = np.zeros(N + 1)

ys[0] = y_init

for _ in range(num_sims):
    for i in range(1, ts.size):
        t = t_init + (i - 1) * dt
        y = ys[i - 1]
        ys[i] = y + mu(y, t) * dt + sigma(y, t) * dW(dt)
    plt.plot(ts, ys)

plt.xlabel("time (s)")
h = plt.ylabel("y")
h.set_rotation(0)
plt.show()


# new maruyama method

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

sigma = 1.  # Standard deviation.
mu = 10.  # Mean.
tau = .05  # Time constant.



dt = .001  # Time step.
T = 1.  # Total time.
n = int(T / dt)  # Number of time steps.
t = np.linspace(0., T, n)  # Vector of times.


sigma_bis = sigma * np.sqrt(2. / tau)
sqrtdt = np.sqrt(dt)

x = np.zeros(n)


for i in range(n - 1):
    x[i + 1] = x[i] + dt * (-(x[i] - mu) / tau) + \
        sigma_bis * sqrtdt * np.random.randn()


fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(t, x, lw=2)


ntrials = 10000
X = np.zeros(ntrials)
# We create bins for the histograms.
bins = np.linspace(-2., 14., 100)
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
for i in range(n):
    # We update the process independently for
    # all trials
    X += dt * (-(X - mu) / tau) + \
        sigma_bis * sqrtdt * np.random.randn(ntrials)
    # We display the histogram for a few points in
    # time
    if i in (5, 50, 900):
        hist, _ = np.histogram(X, bins=bins)
        ax.plot((bins[1:] + bins[:-1]) / 2, hist,
                {5: '-', 50: '.', 900: '-.', }[i],
                label=f"t={i * dt:.2f}")
    ax.legend()

def u(x_p,y_p):
    return 0
# Euler Maruyama method for just the x position
x_p = 0 # x position
y_p = 0 # y postion

new_x_p = x_p+u(x_p,y_p)*h+math.sqrt(2*D*h*epsilon_x)

mean = 0
variance = 1


#dXp = u(Xp,Yp)*dt+math.sqrt(2*D)*dWx
#dYp = v(Xp,Yp)*dt+math.sqrt(2*D)*dWy
#Xp_i+1 = Xp_i+u(Xp_i,Yp_i)*h+math.sqrt(2*D*h*epsilon_x)
#Yp_i+1 = Yp_i+v(Xp_i,Yp_i)*h+math.sqrt(2*D*h*epsilon_y)


    
# Reading the velocity file
#with open('velocityCMM3.dat') as f:
    #for line in f:         # for loop on file object returns one line at a time
        #spl = line.split() # split the line at whitespaces, str.split returns a list
        #lis.append(spl[0]) # append the first item to the output list, use int() get an integer
    #contents = f.read()
    #print(contents)
    #print(lis)
    
#Calculation of position of every particle 
#for p in range(1,Np):
    #Calculations here
    
# This is the uniform random distribution that we will need
#np.random.uniform(1,2) #the range is the size of the computational 2D domain in the x and y directions


# This is the Euler Maruyama method
# num_sims = 30 # Display five runs
# Xp_i = 0
# Yp_i = 0
# u = 0
# v = 0
# t_init = 3
# t_end  = 15
# N      = 1000  # Compute 1000 grid points
# h = t_end-t_init
# dt     = float(t_end - t_init) / N
# y_init = 0
# c_theta = 0.7
# c_mu    = 1.5
# c_sigma = 0.06

# def mu(y, t):
#     """Implement the Ornstein–Uhlenbeck mu."""  # = \theta (\mu-Y_t)
#     return c_theta * (c_mu - y)

# def sigma(y, t):
#     """Implement the Ornstein–Uhlenbeck sigma."""  # = \sigma
#     return c_sigma

# def dW(delta_t):
#     """Sample a random number at each call."""
#     return np.random.normal(loc=0.0, scale=np.sqrt(delta_t))

# ts = np.arange(t_init, t_end + dt, dt)
# ys = np.zeros(N + 1)

# ys[0] = y_init

# for _ in range(num_sims):
#     for i in range(1, ts.size):
#         t = t_init + (i - 1) * dt
#         y = ys[i - 1]
#         ys[i] = y + mu(y, t) * dt + sigma(y, t) * dW(dt)
#     plt.plot(ts, ys)

# plt.xlabel("time (s)")
# h = plt.ylabel("y")
# h.set_rotation(0)
# plt.show()















