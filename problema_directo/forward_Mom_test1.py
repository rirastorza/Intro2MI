#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Solución con MoM de onda plana en 2D y propagación en medio con perdidas
#problema directo usando convolución circular y fft

import numpy as np
from scipy import constants as S
from scipy import special
import matplotlib.pyplot as plt

pi = S.pi
eps0 = S.epsilon_0
c = S.c
mu0 = S.mu_0


freq = 400e6
epsr_b = 30.0
sigma_b = 0.5
epsrCb = epsr_b + sigma_b/(2j*pi*freq*eps0)
k = 2*pi*freq*((epsrCb)**0.5)/c

#landa = c/freq # the wavelength in the air
#k0 = 2*pi/landa # the wave number in the air
#imp = 120*pi # impedance of air
#size_DOI = 2 # size of DOI
#Ni = 16 # number of incidence
#Ns = 32 # number of receiving antennas
#theta = np.linspace(0,2*pi-2*pi/Ni, num=Ni, endpoint=True) # angle of incidence
#phi = 2*pi*np.linspace(0,(Ns-1)/Ns,num=Ns, endpoint=True) # 1 x Ns | angle of receiving antennas
#R_obs = 3 # radius of the circle formed by receiving antennas
#X = R_obs*np.cos(phi) # 1 x Ns % x coordinates of receiving antennas
#Y = R_obs*np.sin(phi) # 1 x Ns % y coordinates of receiving antennas
#epsono_r_c = 2 # the constant relative permittivity of the object


#def A(J,Z,M,landa,epsono_r):
    ###Copyright 2018,  National University of Singapore, Tiantian Yin  
    ### for ii = 1:size(J,2);
    ### temp1 = ifft2(fft2(Z).*fft2(reshape(J(:,ii),M,M),2*M-1,2*M-1));
    ### temp2 = temp1(1:M,1:M);
    ### opa1(:,ii) = J(:,ii)+((1i*2*pi/lambda/(120*pi)*(epsono_r(:)-1))).*reshape(temp2,M^2,1);
    ### end
    #Ni = J.shape[1]
    #J = J.reshape(M,M,Ni)
    #Z = Z[:,:,np.newaxis]#cuidado! Para que funcione de manera equivalente tile y repmat
    #Z = np.tile(Z,(1,1,Ni))
    #opa = np.fft.ifft2(np.fft.fft2(Z,axes = (0,1))*np.fft.fft2(J,(2*M-1,2*M-1),(0,1)),axes = (0,1))
    #opa = opa[0:M,0:M,:]
    #opa = opa.reshape((M**2,Ni))
    #opa = J.reshape((M**2,Ni))+(1j*2*np.pi/landa/(120*np.pi))*(np.tile(epsono_r.T.flatten().reshape((M**2,1)),(1,Ni))-1)*opa
    
    #return opa



#def AH(J,Z,M,landa,epsono_r):
    ###Copyright 2018,  National University of Singapore, Tiantian Yin  
    ### for ii = 1:size(J,2);
    ### temp1 = ifft2(fft2(Z).*fft2(reshape(J(:,ii),M,M),2*M-1,2*M-1));
    ### temp2 = temp1(1:M,1:M);
    ### opa1(:,ii) = J(:,ii)+((1i*2*pi/lambda/(120*pi)*(epsono_r(:)-1))).*reshape(temp2,M^2,1);
    ### end
    #Ni = J.shape[1]
    #J = J.reshape(M,M,Ni)
    #Z = Z[:,:,np.newaxis]#cuidado! Para que funcione de manera equivalente tile y repmat
    #Z = np.tile(Z,(1,1,Ni))
    #opa = np.fft.ifft2(np.fft.fft2(np.conj(Z),axes = (0,1))*np.fft.fft2(J,(2*M-1,2*M-1),(0,1)),axes = (0,1))
    #opa = opa[0:M,0:M,:]
    #opa = opa.reshape((M**2,Ni))
    #opa = J.reshape((M**2,Ni))+np.conj((1j*2*np.pi/landa/(120*np.pi))*(np.tile(epsono_r.T.flatten().reshape((M**2,1)),(1,Ni))-1))*opa
    
    #return opa




##Positions of the cells 
#M = 40 # the square containing the object has a dimension of MxM
#d = size_DOI/M #the nearest distance of two cell centers
#print('landa: ',landa/(epsono_r_c)**.5)
#print('d: ',d)

#tx = d*np.linspace(-(M-1)/2,(M-1)/2,num=M,endpoint = True) #((-(M-1)/2):1:((M-1)/2))*d # 1 x M
#ty = d*np.linspace((M-1)/2,-(M-1)/2,num=M,endpoint = True) #((-(M-1)/2):1:((M-1)/2))*d # 1 x M
#x, y = np.meshgrid(tx, ty)# M x M
#celldia = 2*np.sqrt(d**2/pi) # diameter of cells
#cellrad = celldia/2 #radius of cells

##Relative permittivity of each cell
#epsono_r = np.ones((M,M))
#epsono_r[(x-0.3)**2+(y-0.6)**2 <= 0.2**2] = epsono_r_c
#epsono_r[(x+0.3)**2+(y-0.6)**2<=0.2**2] = epsono_r_c
#epsono_r[(x**2+(y+0.2)**2>=0.3**2) & (x**2+(y+0.2)**2<=0.6**2)] = epsono_r_c

#X_dif,Y_dif = np.meshgrid(d*np.linspace(1-M,M-1,num=2*M-1,endpoint = True),d*np.linspace(1-M,M-1,num=2*M-1,endpoint = True))
#R = np.sqrt(X_dif**2+Y_dif**2) # (2M-1) x (2M-1)

#ZZ = -imp*pi*cellrad/2*special.jv(1,k0*cellrad)*special.hankel1(0,k0*R) #(2M-1) x (2M-1)
#ZZ[M-1,M-1] = -imp*pi*cellrad/2*special.hankel1(1,k0*cellrad)-1j/(2*pi/landa/(imp)) # 1 x 1

#Z = np.zeros((2*M-1,2*M-1),dtype = complex)
#Z[:M,:M] = ZZ[(M-1):(2*M-1),(M-1):(2*M-1)]
#Z[M:(2*M-1),M:(2*M-1)] = ZZ[:(M-1),:(M-1)]
#Z[:M,M:(2*M-1)] = ZZ[M-1:(2*M-1),:(M-1)]
#Z[M:(2*M-1),:M] = ZZ[:(M-1),(M-1):(2*M-1)]

#print(Z[3,0])

##Incident wave (ONDA PLANA)
#E_inc = np.exp(np.matmul((1j*k0*x.T.flatten()).reshape((M**2,1)),(np.cos(theta.T.flatten())).T.reshape((1,Ni)))+np.matmul((1j*k0*y.T.flatten()).reshape((M**2,1)),(np.sin(theta.T.flatten())).T.reshape((1,Ni))))# M^2 x Ni

#print(E_inc.shape)

#b = (-1j*2*pi/(landa*imp))*np.tile((epsono_r.T.flatten()-1).reshape((M**2,1)),(1,Ni))*E_inc # M^2 x Ni

##Using conjugate-gradient method
#np.random.seed(0)
##Jo = np.random.randn(M**2,Ni) # M^2 x Ni
#Jo = 0.1*np.ones((M**2,Ni))
#go = AH(A(Jo,Z,M,landa,epsono_r)-b,Z,M,landa,epsono_r)
#po = -go


#for n in range(20):
    #alphao = -np.sum(np.conj(A(po,Z,M,landa,epsono_r))*(A(Jo,Z,M,landa,epsono_r)-b),axis=0)/np.linalg.norm(A(po,Z,M,landa,epsono_r).reshape((M**2*Ni,1)))**2 # 1 x Ni
    #J = Jo+np.tile(alphao,(M**2,1))*po # M^2 x Ni
    #g = AH(A(J,Z,M,landa,epsono_r)-b,Z,M,landa,epsono_r)# % M^2 x Ni
    #betao = np.sum(np.conj(g)*(g-go),axis = 0)/np.sum(abs(go)**2,axis = 0)# 1 x Ni
    #p = -g+np.tile(betao,(M**2,1))*po#  M^2 x N

    #po = p # M^2 x Ni
    #Jo = J # M^2 x Ni
    #go = g # M^2 x Ni


## Generate Scatterd E field
## We assume that the scattered field is measured at the circle which is
## centered at the original point with a radius equal to 3
##
#X_obs = np.tile(X.T,(M*M,1)).T# Ns x M^2
#Y_obs = np.tile(Y.T,(M*M,1)).T# Ns x M^2

#R = np.sqrt((X_obs-np.tile(x.reshape((M*M,1),order = 'F').T,(Ns,1)))**2+(Y_obs-np.tile(y.reshape((M*M,1),order = 'F').T,(Ns,1)))**2) # Ns x M^2


#ZZ = -imp*np.pi*cellrad/2*special.jv(1,k0*cellrad)*special.hankel1(0,k0*R)#Ns x M^2

#E_s = np.matmul(ZZ,J)# Ns x Ni


#plt.figure()
#extent2=[-1,1,-1,1]
#plt.imshow(epsono_r,extent = extent2,origin='lower')#cmap = 'binary')
##plt.plot(xS,yS,'ow')
#plt.colorbar()

#plt.figure()
##extent2=[-0.25/2,0.25/2,-0.25/2,0.25/2]
#plt.imshow(abs(E_s),cmap = 'pink')#origin='lower')#,extent = extent2)#cmap = 'binary')
##plt.plot(xS,yS,'ow')
#plt.colorbar()
#print(E_s[13,4:8])
#plt.show()

###%
###%
###%nl = 0; % noise level || eg: when noise level is 10%, nl = 0.1.
###%rand_real = randn(Ns,Ni);
###%rand_imag = randn(Ns,Ni);
###%E_Gaussian = 1/sqrt(2) *sqrt(1/Ns/Ni)*norm(E_s,'fro') *nl*(rand_real +1i*rand_imag);
###%E_s = E_s + E_Gaussian;
