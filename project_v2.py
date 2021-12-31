import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.interpolate import interp2d

N = 10000
xmin = -1
xmax = 1
ymin = -1
ymax = 1

mean = 0
variance = 1
D = 0.01
h = 5e-3

tmax = 0.2
ts = np.arange(0, tmax + h, h)

def get_data(path):
    data = np.loadtxt(path)
    X = data[:, 0]
    Y = data[:, 1]
    u = data[:, 2]
    v = data[:, 3]
    return X, Y, u, v

def epsilon(mean=mean, variance=variance, N=N):
    return np.sqrt(2 * D * h) * np.random.normal(mean, np.sqrt(variance), N)

def get_sample(N=N, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax):
    x = np.random.uniform(xmin, xmax, size=N)
    y = np.random.uniform(ymin, ymax, size=N)
    labels = np.zeros(N)
    labels = np.where(x**2 + y**2 < (0.3 * (xmax-xmin) / 2) ** 2, 1, 0)
    
    return x, y, labels

def plot(x, y, labels, t, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax):
    fig = plt.figure(figsize=(8, 8))
    plt.grid( linestyle='--', linewidth=1)
    plt.scatter(x[labels==0], y[labels==0], color='red', s=2)
    plt.scatter(x[labels==1], y[labels==1], color='blue', s=2)
    plt.colorbar()
    plt.axis([xmin, xmax, ymin, ymax])
    plt.xlabel("x - axis")
    plt.ylabel("y - axis")
    plt.title(f't={t}')
    return plt.show()

def plot_on_grid(x, y, labels, t, ax, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax):
    #fig = plt.figure(figsize=(8, 8))
    ax.grid( linestyle='--', linewidth=1)
    im = ax.scatter(x[labels==0], y[labels==0], color='red', s=2)
    ax.scatter(x[labels==1], y[labels==1], color='blue', s=2)
    #ax.colorbar()
    plt.colorbar(im, ax=ax)
    ax.axis([xmin, xmax, ymin, ymax])
    ax.set_xlabel("x - axis")
    ax.set_ylabel("y - axis")
    ax.set_title(f't={t}')
    return #plt.show()

def u_f(x, y):
    return u_interpolate(x, y)

def v_f(x, y):
    return v_interpolate(x, y)

def Euler_Maruyama(x, y, h=h):
    # x position
    vx = np.array([u_f(x[i], y[i]) for i in range(N)]).ravel()
    #print(vx)
    x = x +  vx * h + epsilon()    
    #right boundary
    x = np.where(x > xmax, 2 * xmax - x, x)
    #left boundary
    x = np.where(x < xmin, 2 * xmin - x, x)
    
    # y position
    vy = np.array([v_f(x[i], y[i]) for i in range(N)]).ravel()
    y = y + vy * h + epsilon()    
    #up boundary
    y = np.where(y > ymax, 2 * ymax - y, y)
    #down boundary
    y = np.where(y < ymin, 2 * ymin - y, y)
    return x, y




X, Y, u, v = get_data('velocityCMM3.dat')

u_interpolate = interp2d(X, Y, u,'linear')
v_interpolate = interp2d(X, Y, v,'linear')

x, y, labels = get_sample(N=N, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

gs = gridspec.GridSpec(2, 2, hspace = 0.2, wspace=0.2)
fig = plt.figure(figsize=(15, 10))

for i, t in enumerate(ts):
    if t % 0.1 == 0:
        ax = fig.add_subplot(gs[int(t//0.1)])
        plot_on_grid(x, y, labels, t=t, ax=ax)
    
    x, y = Euler_Maruyama(x, y, h=h)
    
Nx = 64
Ny = 64
all_particles, xedges, yedges = np.histogram2d(x, y, range=[(-1, 1), (-1, 1)], bins=[Nx, Ny])
impurity_particles, _, _ = np.histogram2d(x[labels==1], y[labels==1], range=[(-1, 1), (-1, 1)], bins=[Nx, Ny])

all_particles = np.where(all_particles==0, 1, all_particles) # to avoid zero dividing
concentration = impurity_particles / all_particles

x_grid, y_grid = np.meshgrid(xedges, yedges)
if Ny != 1:
    
    fig = plt.figure(figsize=(6, 6))

    ax = plt.pcolormesh(x_grid, y_grid, concentration, cmap='rainbow')
    fig.colorbar(ax)
else:
    plt.plot(xedges[1:], concentration, 'o-')
    
# 1D PROBLEM
Nx = 64
Ny = 1
N_particles = [1024, 8192]
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for i_np, Np in enumerate(N_particles):
    axes[i_np].set_xlabel('$x$')
    for i in range(3):
        x, y, labels = get_sample(N=Np, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
        for t in ts:
            x, y = Euler_Maruyama(x, y, h=h)
        all_particles, xedges, yedges = np.histogram2d(x, y, range=[(-1, 1), (-1, 1)], bins=[Nx, Ny])
        impurity_particles, _, _ = np.histogram2d(x[labels==1], y[labels==1], range=[(-1, 1), (-1, 1)], bins=[Nx, Ny])
        concentration = (impurity_particles / all_particles).ravel()
        axes[i_np].plot(xedges[1:], concentration, 'o-', label=f'$N_p$={Np}, run {i+1}')
    axes[i_np].legend()