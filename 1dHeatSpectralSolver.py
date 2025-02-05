# Write your code here :-)
import numpy as np
import matplotlib.pyplot as plt

#Numerical Parameters
L = 2*np.pi #Length of spatial interval
N = 2**10 #Number of grid points
T = 1.0 #Length of simulation
dt = 0.01 #Size of timesteps
Nt = int(T/dt) #Number of timesteps

#Physical Parameters
alpha = 10 #Thermal Diffusivity

#Define/Set IC
def initial_condition(x):
    return np.cos(x)

#Define spectral solver
def spectral_heat_solver1d(L,N,T,Nt):
    x = np.linspace(0,L,num=N,endpoint=False) #Generate mesh
    k = np.fft.fftfreq(N,d=L/N)*2*np.pi*(1/L) #Generate relevant wavenumbers
    u0 = initial_condition(x)

    u0_hat = np.fft.fft(u0) #Obtain Fourier coefficients

    for j in range(1,Nt+1):
        u_hat = np.exp(-alpha*(k**2)*j*dt)*u0_hat

    return x, np.fft.ifft(u_hat).real

#Run solver and return plot of solution
x, u_sol = spectral_heat_solver1d(L,N,T,Nt)
plt.plot(x, u_sol, label='Solution')
plt.xlim(0,2*np.pi)
plt.ylim(-1,1)
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.title('Spectral Solution of 1+1d Heat Equation')
plt.show()

