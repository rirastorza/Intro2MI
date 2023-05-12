#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Onda CILÍNDRICA INCIDENTE en 2D y propagación en medio con perdidas
#SOLUCIÓN ANALÍTICA

import numpy as np
from scipy import constants as S
from scipy import special
import matplotlib.pyplot as plt

pi = S.pi
epsilon0 = S.epsilon_0
c = S.c
mu0 = S.mu_0


freq = 1.0e9
frel = 20e9
sigmadc = 800e-3
epsb = 4.8+(77.0-4.8)/(1+1j*freq/frel)+sigmadc/(1j*epsilon0*2*pi*freq)
kb = 2 * pi * freq * (mu0*epsilon0*epsb)**0.5

size_DOI = 0.25 # size of DOI
Ni = 8 # number of incidence
Ns = 8 # number of receiving antennas
theta = np.linspace(0,2*pi-2*pi/Ni, num=Ni, endpoint=True) # angle of incidence
phi = 2*pi*np.linspace(0,(Ns-1)/Ns,num=Ns, endpoint=True) # 1 x Ns | angle of receiving antennas
R_obs = 0.075 # radius of the circle formed by receiving antennas
X = R_obs*np.cos(phi) # 1 x Ns % x coordinates of receiving antennas
Y = R_obs*np.sin(phi) # 1 x Ns % y coordinates of receiving antennas

M = 100 # the square containing the object has a dimension of MxM
d = size_DOI/M #the nearest distance of two cell centers
tx = d*np.linspace(-(M-1)/2,(M-1)/2,num=M,endpoint = True) #((-(M-1)/2):1:((M-1)/2))*d # 1 x M
ty = d*np.linspace((M-1)/2,-(M-1)/2,num=M,endpoint = True) #((-(M-1)/2):1:((M-1)/2))*d # 1 x M
x, y = np.meshgrid(tx, ty)# M x M


#Einc CILÍNDRICO implementado basado en (5-119) y (5-103) en [Harrington2001]
def cart2pol(x,y):
    rho = (x**2.0+y**2.0)**0.5
    phi = np.arctan2(y,x)
    #phi = N.arctan(y/x)
    return phi,rho

print('Fuente: ',R_obs, theta.T.flatten())
rho_s = R_obs
phi_s = theta.T.flatten()
phi, rho = cart2pol(x.T.flatten(),y.T.flatten())
absrho = np.zeros((M**2,Ni),dtype = complex)
for mm,angulo in enumerate(phi_s):
    absrho[:,mm] = (rho**2.+rho_s**2.-2.0*rho*rho_s*np.cos(phi-angulo))**0.5
    
#Cuidado: ver expresión Ecuación siguiente a Ec. (5-102) en Harrington
#con I = 1 (fuente de corriente) o pag 60 Pastorino
E_inc = -2*pi*freq*mu0/4*special.hankel2(0,kb*absrho)

print(E_inc.shape)

plt.figure(1)
extent2=[-0.25/2,0.25/2,-0.25/2,0.25/2]
plt.imshow(abs(E_inc[:,0].reshape((M,M))).T,extent = extent2)

plt.figure(2)

plt.plot(tx,abs(E_inc[:,0].reshape((M,M)))[:,int(M/2)])

plt.show()

