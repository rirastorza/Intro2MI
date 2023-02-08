#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Solución con MoM de onda CILÍNDRICA en 2D y propagación en medio con perdidas
#problema directo usando convolución circular y fft

import numpy as np
from scipy import constants as S
from scipy import special
import matplotlib.pyplot as plt

pi = S.pi
eps0 = S.epsilon_0
c = S.c
mu0 = S.mu_0


freq = 1100e6
epsr_b = 40.0#Similar a permitividad de 80%glicerol-agua para f = 400MHz.
sigma_b = 0.5#Similar a conductividad de 80%glicerol-agua para f = 400MHz.
epsrCb = epsr_b + sigma_b/(2j*pi*freq*eps0)
kb = 2*pi*freq*((epsrCb)**0.5)/c
k0 = 2*pi*freq/c # the wave number in the air
landa0 = 2*pi/k0
imp0 = 120*pi # imp0edance of air

size_DOI = 0.25 # size of DOI
Ni = 8 # number of incidence
Ns = 32 # number of receiving antennas
theta = np.linspace(0,2*pi-2*pi/Ni, num=Ni, endpoint=True) # angle of incidence
phi = 2*pi*np.linspace(0,(Ns-1)/Ns,num=Ns, endpoint=True) # 1 x Ns | angle of receiving antennas
R_obs = 0.075 # radius of the circle formed by receiving antennas
X = R_obs*np.cos(phi) # 1 x Ns % x coordinates of receiving antennas
Y = R_obs*np.sin(phi) # 1 x Ns % y coordinates of receiving antennas
sigma_c = 0.4
epsono_r_c = 50.+ sigma_c/(2j*pi*freq*eps0) # the constant relative permittivity of the object


def A(J,Z,M,landa,epsono_r,epsrCb):
    ##Copyright 2018,  National University of Singapore, Tiantian Yin  
    ## for ii = 1:size(J,2);
    ## temp1 = ifft2(fft2(Z).*fft2(reshape(J(:,ii),M,M),2*M-1,2*M-1));
    ## temp2 = temp1(1:M,1:M);
    ## opa1(:,ii) = J(:,ii)+((1i*2*pi/lambda/(120*pi)*(epsono_r(:)-1))).*reshape(temp2,M^2,1);
    ## end
    Ni = J.shape[1]
    J = J.reshape(M,M,Ni)
    Z = Z[:,:,np.newaxis]#cuidado! Para que funcione de manera equivalente tile y repmat
    Z = np.tile(Z,(1,1,Ni))
    opa = np.fft.ifft2(np.fft.fft2(Z,axes = (0,1))*np.fft.fft2(J,(2*M-1,2*M-1),(0,1)),axes = (0,1))
    opa = opa[0:M,0:M,:]
    opa = opa.reshape((M**2,Ni))
    opa = J.reshape((M**2,Ni))+(1j*2*pi/landa0/imp0)*(np.tile(epsono_r.T.flatten().reshape((M**2,1)),(1,Ni))-epsrCb)*opa
    
    return opa



def AH(J,Z,M,landa,epsono_r,epsrCb):
    ##Copyright 2018,  National University of Singapore, Tiantian Yin  
    ## for ii = 1:size(J,2);
    ## temp1 = ifft2(fft2(Z).*fft2(reshape(J(:,ii),M,M),2*M-1,2*M-1));
    ## temp2 = temp1(1:M,1:M);
    ## opa1(:,ii) = J(:,ii)+((1i*2*pi/lambda/(120*pi)*(epsono_r(:)-1))).*reshape(temp2,M^2,1);
    ## end
    Ni = J.shape[1]
    J = J.reshape(M,M,Ni)
    Z = Z[:,:,np.newaxis]#cuidado! Para que funcione de manera equivalente tile y repmat
    Z = np.tile(Z,(1,1,Ni))
    opa = np.fft.ifft2(np.fft.fft2(np.conj(Z),axes = (0,1))*np.fft.fft2(J,(2*M-1,2*M-1),(0,1)),axes = (0,1))
    opa = opa[0:M,0:M,:]
    opa = opa.reshape((M**2,Ni))
    opa = J.reshape((M**2,Ni))+np.conj((1j*2*pi/landa/imp0)*(np.tile(epsono_r.T.flatten().reshape((M**2,1)),(1,Ni))-epsrCb))*opa
    
    return opa


#Positions of the cells 
M = 150 # the square containing the object has a dimension of MxM
d = size_DOI/M #the nearest distance of two cell centers
print('landa: ',landa0/(epsono_r_c.real)**.5)
print('d: ',d)

tx = d*np.linspace(-(M-1)/2,(M-1)/2,num=M,endpoint = True) #((-(M-1)/2):1:((M-1)/2))*d # 1 x M
ty = d*np.linspace((M-1)/2,-(M-1)/2,num=M,endpoint = True) #((-(M-1)/2):1:((M-1)/2))*d # 1 x M
x, y = np.meshgrid(tx, ty)# M x M
celldia = 2*np.sqrt(d**2/pi) # diameter of cells
cellrad = celldia/2 #radius of cells

#Relative permittivity of each cell
r_cilinder = 25.0e-3# 50.0e-3
epsono_r = epsrCb*np.ones((M,M),dtype = complex)
epsono_r[(x-0.0)**2+(y-0.0)**2 <= r_cilinder**2] = epsono_r_c
#epsono_r[(x+0.3)**2+(y-0.6)**2<=0.2**2] = epsono_r_c
#epsono_r[(x**2+(y+0.2)**2>=0.3**2) & (x**2+(y+0.2)**2<=0.6**2)] = epsono_r_c

X_dif,Y_dif = np.meshgrid(d*np.linspace(1-M,M-1,num=2*M-1,endpoint = True),d*np.linspace(1-M,M-1,num=2*M-1,endpoint = True))
R = np.sqrt(X_dif**2+Y_dif**2) # (2M-1) x (2M-1)

ZZ = -imp0*pi*cellrad/2*special.jv(1,kb*cellrad)*special.hankel1(0,kb*R) #(2M-1) x (2M-1)
ZZ[M-1,M-1] = -imp0*pi*cellrad/2*special.hankel1(1,kb*cellrad)-1j/(2*pi/landa0/(imp0)) # 1 x 1

Z = np.zeros((2*M-1,2*M-1),dtype = complex)
Z[:M,:M] = ZZ[(M-1):(2*M-1),(M-1):(2*M-1)]
Z[M:(2*M-1),M:(2*M-1)] = ZZ[:(M-1),:(M-1)]
Z[:M,M:(2*M-1)] = ZZ[M-1:(2*M-1),:(M-1)]
Z[M:(2*M-1),:M] = ZZ[:(M-1),(M-1):(2*M-1)]


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
#print(phi.shape,rho.shape,E_inc.shape)

#plt.figure()
#plt.imshow(abs(E_inc[:,2].reshape((M,M))),cmap = 'pink')#origin='lower')#,extent = extent2)#cmap = 'binary')
#plt.colorbar()


##E_inc = np.exp(np.matmul((1j*kb*x.T.flatten()).reshape((M**2,1)),(np.cos(theta.T.flatten())).T.reshape((1,Ni)))+np.matmul((1j*kb*y.T.flatten()).reshape((M**2,1)),(np.sin(theta.T.flatten())).T.reshape((1,Ni))))# M^2 x Ni

b = (-1j*2*pi/(landa0*imp0))*np.tile((epsono_r.T.flatten()-epsrCb).reshape((M**2,1)),(1,Ni))*E_inc # M^2 x Ni

#Using conjugate-gradient method
np.random.seed(0)
Jo = np.random.randn(M**2,Ni)+1j*np.random.randn(M**2,Ni) # M^2 x Ni
#Jo = 0.1*np.ones((M**2,Ni))
go = AH(A(Jo,Z,M,landa0,epsono_r,epsrCb)-b,Z,M,landa0,epsono_r,epsrCb)
po = -go

niter = 200
for n in range(niter):
    alphao = -np.sum(np.conj(A(po,Z,M,landa0,epsono_r,epsrCb))*(A(Jo,Z,M,landa0,epsono_r,epsrCb)-b),axis=0)/np.linalg.norm(A(po,Z,M,landa0,epsono_r,epsrCb).reshape((M**2*Ni,1)))**2 # 1 x Ni
    J = Jo+np.tile(alphao,(M**2,1))*po # M^2 x Ni
    g = AH(A(J,Z,M,landa0,epsono_r,epsrCb)-b,Z,M,landa0,epsono_r,epsrCb)# % M^2 x Ni
    betao = np.sum(np.conj(g)*(g-go),axis = 0)/np.sum(abs(go)**2,axis = 0)# 1 x Ni
    p = -g+np.tile(betao,(M**2,1))*po#  M^2 x N

    po = p # M^2 x Ni
    Jo = J # M^2 x Ni
    go = g # M^2 x Ni


# Generate Scatterd E field
# We assume that the scattered field is measured at the circle which is
# centered at the original point with a radius equal to 3
#
X_obs = np.tile(X.T,(M*M,1)).T# Ns x M^2
Y_obs = np.tile(Y.T,(M*M,1)).T# Ns x M^2

R = np.sqrt((X_obs-np.tile(x.reshape((M*M,1),order = 'F').T,(Ns,1)))**2+(Y_obs-np.tile(y.reshape((M*M,1),order = 'F').T,(Ns,1)))**2) # Ns x M^2


ZZ = -imp0*pi*cellrad/2*special.jv(1,kb*cellrad)*special.hankel1(0,kb*R)#Ns x M^2

E_s = np.matmul(ZZ,J)# Ns x Ni
print(E_s.shape)

plt.figure()
extent2=[-.25/2,.25/2,-.25/2,.25/2]
plt.imshow(epsono_r.real,extent = extent2,origin='lower')#cmap = 'binary')
#plt.plot(xS,yS,'ow')
plt.colorbar()

plt.figure()
#extent2=[-0.25/2,0.25/2,-0.25/2,0.25/2]
plt.imshow(abs(E_s),cmap = 'pink')#origin='lower')#,extent = extent2)#cmap = 'binary')
#plt.plot(xS,yS,'ow')
plt.colorbar()

plt.figure()
#extent2=[-0.25/2,0.25/2,-0.25/2,0.25/2]
#plt.imshow(abs(E_s),cmap = 'pink')#origin='lower')#,extent = extent2)#cmap = 'binary')
plt.plot(abs(E_s)[:,0],'o-')
#plt.colorbar()

np.savez('test_cil_M_'+str(M)+'_niter_'+str(niter), Es=E_s)

plt.show()

