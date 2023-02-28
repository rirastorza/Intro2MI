#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Simulación del problema inverso utilizando el método Backpropagation

import numpy as np
from scipy import constants as S
from scipy import special
import matplotlib.pyplot as plt

#-------------------------
#idem forward_Mom_test0.py
pi = S.pi
eps0 = S.epsilon_0
c = S.c
mu0 = S.mu_0

freq = 400e6
landa = c/freq # the wavelength in the air
k0 = 2*pi/landa # the wave number in the air
imp = 120*pi # impedance of air
size_DOI = 2 # size of DOI
Ni = 16 # number of incidence
Ns = 32 # number of receiving antennas
theta = np.linspace(0,2*pi-2*pi/Ni, num=Ni, endpoint=True) # angle of incidence
phi = 2*pi*np.linspace(0,(Ns-1)/Ns,num=Ns, endpoint=True) # 1 x Ns | angle of receiving antennas
R_obs = 3 # radius of the circle formed by receiving antennas
X = R_obs*np.cos(phi) # 1 x Ns % x coordinates of receiving antennas
Y = R_obs*np.sin(phi) # 1 x Ns % y coordinates of receiving antennas
#--------------------------



def Gd(J,Z,M):
    #% for ii = 1:size(J,2);
    #% temp1 = ifft2(fft2(Z).*fft2(reshape(J(:,ii),M,M),2*M-1,2*M-1));
    #% opa1(:,ii) = reshape(temp1(1:M,1:M),M^2,1);
    #% end

    Ni = J.shape[1]
    J = J.reshape(M,M,Ni)
    Z = Z[:,:,np.newaxis]#cuidado! Para que funcione de manera equivalente tile y repmat
    Z = np.tile(Z,(1,1,Ni))
    opa = np.fft.ifft2(np.fft.fft2(Z,axes = (0,1))*np.fft.fft2(J,(2*M-1,2*M-1),(0,1)),axes = (0,1))
    opa = opa[0:M,0:M,:]
    opa = opa.reshape((M**2,Ni))
    
    return opa

 



#Parameters (keep unchanged in the inverse problem)
#Position of the cells
M = 64 # the square DOI has a dimension of MxM % This value of M is set to be smaller than the value of M used in forward problem to avoid "inverse crime". 
d = size_DOI/M #the nearest distance of two cell centers
tx = d*np.linspace(-(M-1)/2,(M-1)/2,num=M,endpoint = True) #((-(M-1)/2):1:((M-1)/2))*d # 1 x M
ty = d*np.linspace((M-1)/2,-(M-1)/2,num=M,endpoint = True) #((-(M-1)/2):1:((M-1)/2))*d # 1 x M
x, y = np.meshgrid(tx, ty)# M x M
celldia = 2*np.sqrt(d**2/pi) # diameter of cells
cellrad = celldia/2 #radius of cells
print(cellrad)
# used to do circular convolution with the eletric diple to generate the
# scattered E field within DOI (used for calculation of Gd)
# note that one element in ZZ matrix corresponds to R = 0
X_dif,Y_dif = np.meshgrid(d*np.linspace(1-M,M-1,num=2*M-1,endpoint = True),d*np.linspace(1-M,M-1,num=2*M-1,endpoint = True))
R = np.sqrt(X_dif**2+Y_dif**2) # (2M-1) x (2M-1)


ZZ = -imp*pi*cellrad/2*special.jv(1,k0*cellrad)*special.hankel1(0,k0*R) #(2M-1) x (2M-1)
ZZ[M-1,M-1] = -imp*pi*cellrad/2*special.hankel1(1,k0*cellrad)-1j/(2*pi/landa/(imp)) # 1 x 1

Z = np.zeros((2*M-1,2*M-1),dtype = complex)
Z[:M,:M] = ZZ[(M-1):(2*M-1),(M-1):(2*M-1)]
Z[M:(2*M-1),M:(2*M-1)] = ZZ[:(M-1),:(M-1)]
Z[:M,M:(2*M-1)] = ZZ[M-1:(2*M-1),:(M-1)]
Z[M:(2*M-1),:M] = ZZ[:(M-1),(M-1):(2*M-1)]


# Calculation of Gs 
X_obs = np.tile(X.T,(M*M,1)).T# Ns x M^2
Y_obs = np.tile(Y.T,(M*M,1)).T# Ns x M^2

R = np.sqrt((X_obs-np.tile(x.reshape((M*M,1),order = 'F').T,(Ns,1)))**2+(Y_obs-np.tile(y.reshape((M*M,1),order = 'F').T,(Ns,1)))**2) # Ns x M^2



Gs = -imp*np.pi*cellrad/2*special.jv(1,k0*cellrad)*special.hankel1(0,k0*R)#Ns x M^2

print()
#Incident wave (ONDA PLANA)
E_inc = np.exp(np.matmul((1j*k0*x.T.flatten()).reshape((M**2,1)),(np.cos(theta.T.flatten())).T.reshape((1,Ni)))+np.matmul((1j*k0*y.T.flatten()).reshape((M**2,1)),(np.sin(theta.T.flatten())).T.reshape((1,Ni))))# M^2 x Ni



##-----------------------------------------------------

#M = 40
npzfile = np.load('test_Inv_M_'+str(40)+'.npz')
E_s = npzfile['Es']
#print(Gs[:,0])
#gamma = sum(E_s.*conj(Gs*Gs'*E_s),1)./sum(abs(Gs*Gs'*E_s).^2,1); % 1 x Ni
gamma = np.sum(E_s*np.conj(np.matmul(np.matmul(Gs,np.conj(Gs).T),E_s)),axis=0)/np.sum(abs(np.matmul(np.matmul(Gs,np.conj(Gs).T),E_s))**2,axis=0) # 1 x Ni



J = np.tile(gamma,(M**2,1))*np.matmul(np.conj(Gs).T,E_s)#M^2 x Ni
Et = E_inc+Gd(J,Z,M)# M^2 x 1

Gsprint = Gs.T
print(Gsprint[:,0])

num = np.sum(J*np.conj(Et),axis=1)
den = np.sum(np.conj(Et)*Et,axis=1)

chai = (num/den).reshape((M,M))# M x M

plt.figure(1)
#extent2=[-0.25/2,0.25/2,-0.25/2,0.25/2]
plt.imshow(chai.imag,cmap = 'pink',origin='lower')#origin='lower')#,extent = extent2)#cmap = 'binary')
plt.colorbar()

plt.show()

