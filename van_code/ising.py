import torch
import numpy as np

def ising_energy(x,beta):
    energy=x*(torch.roll(x,1,-1)+torch.roll(x,1,-2))
    return -beta*energy.sum((-2,-1))

def local_ising_energy(x):
    return (torch.roll(x,1,-1)+torch.roll(x,-1,-1)+torch.roll(x,1,-2)+torch.roll(x,-1,-2))

def analytical_solution(beta,L):
    logz0=np.log((2*np.sinh(2*beta)))*(L**2/2)-np.log(2)
    z_tmp=Z1(beta,L)+Z2(beta,L)+Z3(beta,L)+Z4(beta,L)
    #print(Z1(beta,L),Z2(beta,L),Z3(beta,L),Z4(beta,L))
    logz=logz0+np.log(z_tmp)
    return -logz

def Z1(beta,L):
    z=1
    for r in range(0,L):
        z*=2.0*np.cosh(0.5*L*gamma_r(beta,L,2.0*r+1.0))
    return z

def Z2(beta,L):
    z=1
    for r in range(0,L):
        z*=2.0*np.sinh(0.5*L*gamma_r(beta,L,2.0*r+1.0))
    return z

def Z3(beta,L):
    z=1
    for r in range(0,L):
        z*=2.0*np.cosh(0.5*L*gamma_r(beta,L,2.0*r))
    return z

def Z4(beta,L):
    z=1
    for r in range(0,L):
        z*=2.0*np.sinh(0.5*L*gamma_r(beta,L,2.0*r))
    return z

def gamma_r(beta,L,r):
    if r==0:
        return 2.0*beta + np.log(np.tanh(beta))
    else:
        return np.log(c_R(beta,r,L)+np.sqrt(c_R(beta,r,L)**2-1.0))

def c_R(beta,r,L):
    return np.cosh(2.0*beta)*coth(2.0*beta)-np.cos(r*np.pi/L)

def coth(x):
    return np.cosh(x)/np.sinh(x)
