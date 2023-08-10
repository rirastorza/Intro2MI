#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Solución con MoM de onda plana en 2D y propagación en vacio
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
landa = c/freq # the wavelength in the air
k0 = 2*pi/landa # the wave number in the air
imp = 120*pi # impedance of air
size_DOI = 2 # size of DOI
Ni = 1 # number of incidence
Ns = 16 # number of receiving antennas
theta = np.linspace(0,2*pi-2*pi/Ni, num=Ni, endpoint=True) # angle of incidence
phi = 2*pi*np.linspace(0,(Ns-1)/Ns,num=Ns, endpoint=True) # 1 x Ns | angle of receiving antennas
R_obs = 3 # radius of the circle formed by receiving antennas
X = R_obs*np.cos(phi) # 1 x Ns % x coordinates of receiving antennas
Y = R_obs*np.sin(phi) # 1 x Ns % y coordinates of receiving antennas
epsono_r_c = 1.8 # the constant relative permittivity of the object

#Positions of the cells 
M = 40 # the square containing the object has a dimension of MxM
d = size_DOI/M #the nearest distance of two cell centers
print('landa: ',landa/(epsono_r_c)**.5)
print('d: ',d)

tx = d*np.linspace(-(M-1)/2,(M-1)/2,num=M,endpoint = True) #((-(M-1)/2):1:((M-1)/2))*d # 1 x M
ty = d*np.linspace((M-1)/2,-(M-1)/2,num=M,endpoint = True) #((-(M-1)/2):1:((M-1)/2))*d # 1 x M
x, y = np.meshgrid(tx, ty)# M x M
celldia = 2*np.sqrt(d**2/pi) # diameter of cells
cellrad = celldia/2 #radius of cells

#Relative permittivity of each cell
epsono_r = np.ones((M,M))
epsono_r[(x)**2+(y)**2 <= 0.8**2] = epsono_r_c

print(x)
print(epsono_r)


cte = (1j/2)*(pi*k0*cellrad*special.hankel2(1,k0*cellrad)-2j)#cuando estoy en la misma celda

matrizCtes = np.zeros((M**2,M**2),dtype = complex)

Xv = x.reshape((M**2,1))
Yv = y.reshape((M**2,1))
ji = epsono_r.reshape((M**2,1))-1
for mm in range(len(matrizCtes)):
    for nn in range(len(matrizCtes)):
        if mm == nn:
            matrizCtes[mm,nn] = (ji[nn])*cte
        else:
            R = np.sqrt((Xv[mm]-Xv[nn])**2+(Yv[mm]-Yv[nn])**2)
            matrizCtes[mm,nn] = (ji[nn])*(1j/2)*pi*k0*cellrad*special.jv(1,k0*cellrad)*special.hankel2(0,k0*R)

np.set_printoptions(precision=3)

D = np.eye(M**2,M**2)

A = D+matrizCtes

#Incident wave (linea de corriente)
absrho = ((x.T.flatten()-6.0)**2+(y.T.flatten()-0)**2)**0.5
E_inc = (-2*pi*freq*mu0/4*special.hankel2(0,k0*absrho)).T.reshape((M**2,1))
#print(E_inc.shape)
b = E_inc.reshape((M**2,1))
#Solución de la ecuación lineal
Et = np.linalg.solve(A, b)

plt.figure(1)
extent2=[-1,1,-1,1]
plt.imshow(epsono_r,extent = extent2,origin='lower')#cmap = 'binary')
#plt.plot(xS,yS,'ow')
plt.colorbar()

plt.figure(2)
#extent2=[-0.25/2,0.25/2,-0.25/2,0.25/2]
plt.imshow(abs(Et.reshape((M,M)).T),cmap ='pink',origin='lower',extent = extent2)#cmap = 'binary')
#plt.plot(xS,yS,'ow')
plt.colorbar()

from forward import *
import os
import numpy as np
from numpy import genfromtxt

f =  0.4e9 #Frecuencia en Hz

#caja de simulacion en [m]
sx = 14#15
sy = 14#15
box = [sx,sy]
a = 0.1#0.1 #meep unit
#Antenas---------------------------------------------------------------
nant_f = 1 #antenas emisoras
rant_f = 6 #radio de antenas fuentes [m]
nant_r = 16 #antenas receptoras
rant_r =3#3 #radio de antenas receptoras[m]
tx = 0 # Fuente analizada
#Medio acoplante---------------------------------------------------------------
epsc = 1
# Medio dispersor (cilindro)---------------------------------------------------
r = 0.8 #10.0e-3/2 #radio del cilindro
Xc = 0    #posición del centro en x
Yc = 0   #posición del centro en y
epsr = 1.8
resolucion = 5 #resolución FDTD

#%%

TRANSMISOR_parameters = TRANSMISOR_parameters()
ACOPLANTE_parameters = ACOPLANTE_parameters()
cilindro = SCATTERER_parameters()

#Antenas-----------------------------------------------------------------------
TRANSMISOR_parameters.f = f #frecuencia 1 GHz (por defecto)
TRANSMISOR_parameters.amp =3750#3500 #Amplitud de la fuente
TRANSMISOR_parameters.rhoS = rant_f #radio de transmisores
TRANSMISOR_parameters.S = nant_f ##cantidad de transmisores (fuentes)
#Coordenadas antenas fuentes
angulo_f = N.linspace(0.0, 2.0*pi, nant_f+1)
xantenas_f = (TRANSMISOR_parameters.rhoS)*N.cos(angulo_f)
yantenas_f = (TRANSMISOR_parameters.rhoS)*N.sin(angulo_f)
RECEPTOR_parameters.f = f #frecuencia 1 GHz (por defecto)
RECEPTOR_parameters.amp =3750#3500 #Amplitud de la fuente
RECEPTOR_parameters.rhoS = rant_r #radio de transmisores
RECEPTOR_parameters.S = nant_r ##cantidad de transmisores (fuentes)
#Coordenadas antenas receptoras
angulo_r = N.linspace(0.0, 2.0*pi, nant_r+1)
xantenas_r= (RECEPTOR_parameters.rhoS)*N.cos(angulo_r)
yantenas_r = (RECEPTOR_parameters.rhoS)*N.sin(angulo_r)
#Medio acoplante---------------------------------------------------------------
#Definición de Debye para simular:
ACOPLANTE_parameters.f = TRANSMISOR_parameters.f
ACOPLANTE_parameters.epsr = epsc.real 
ACOPLANTE_parameters.sigma = -epsc.imag*(eps0*2*pi*TRANSMISOR_parameters.f)
# Medio dispersor (cilindro)---------------------------------------------------
cilindro.epsr = epsr #permitividad relativa. Entre [10.0, 80.0]
cilindro.sigma = 0.0
cilindro.f = TRANSMISOR_parameters.f #frecuencia 1 GHz (por defecto).
cilindro.radio = r
cilindro.xc = Xc
cilindro.yc = Yc

#%% 
#coordenadas receptores en n
nxantenas_r=np.round(xantenas_r*resolucion/a)+sx/a*resolucion/2
nyantenas_r=np.round(yantenas_r*resolucion/a)+sx/a*resolucion/2

fig1 = plt.figure(3)
f1 = fig1.add_subplot(111)
cilindro1 = plt.Circle((Xc,Yc),r, color='g',fill = False) #Dibujo la geometría generada
##en m
plt.xlim(-sx/2, sx/2)
plt.ylim(-sy/2, sy/2)

#f2.set_aspect(1)
f1.add_artist(cilindro1)
f1.plot(xantenas_r,yantenas_r,'ok')
f1.plot(xantenas_f,yantenas_f,'or')
f1.set_aspect('equal')

Ezfdtd,eps_data = RunMeep(cilindro,ACOPLANTE_parameters,TRANSMISOR_parameters, tx, box,RES = resolucion,calibration = False, unit = a)

plt.figure(4)
extent2=[-7,7,-7,7]
plt.imshow(abs(Ezfdtd).T,cmap ='pink',origin='lower',extent = extent2)#cmap = 'binary')
#plt.plot(xS,yS,'ow')
plt.colorbar()


Etmom = Et.reshape((M,M))



#np.savez('test_MoMnuevo_M_'+str(M), Etmom=E_s)

np.savez('test_MoM_M_FDTD'+str(M), Etmom = Etmom,Ezfdtd = Ezfdtd)


###Comparando con octave
##import scipy.io
###save -v7 E_s_for_test.mat E_s
##E_s_mat = scipy.io.loadmat('E_s_for_test.mat')



##Campo disperso en las antenas receptoras
#phi = 2*pi*np.linspace(0,(Ns-1)/Ns,num=Ns, endpoint=True) # 1 x Ns | angle of receiving antennas
#R_obs = 3 # radius of the circle formed by receiving antennas
#xobs = R_obs*np.cos(phi) # 1 x Ns % x coordinates of receiving antennas
#yobs = R_obs*np.sin(phi) # 1 x Ns % y coordinates of receiving antennas
#Es = np.zeros((Ns,1),dtype = complex)
#for n in range(Ns):
    #R = np.sqrt((xobs[n]-x.T.flatten())**2.0+(yobs[n]-y.T.flatten())**2.0)
    #Es[n] = -1j*(pi*k0/2)*np.sum(ji*Et*cellrad*special.jv(1,k0*cellrad)*special.hankel2(0,k0*R))

#print(Es.shape)
###Ahora el campo total
#absrho = ((xobs-6.0)**2+(yobs-0)**2)**0.5
#E_inc = -2*pi*freq*mu0/4*special.hankel2(0,k0*absrho)
#Etobs = Es#+E_inc

tyfdtd = np.linspace(-7,7,num=len(Ezfdtd),endpoint = True)

fig3 = plt.figure(5)
f3 = fig3.add_subplot(211)
f3.plot(ty,np.abs(Etmom[:,int(M/2)]),'o-b')
f3.plot(tyfdtd,np.abs(Ezfdtd[:,int(len(Ezfdtd)/2)]),'-k')
f3.set_xlabel('x')
f3.set_ylabel(r'abs($E_{z}$)')
f3 = fig3.add_subplot(212)
f3.plot(ty,np.angle(Etmom[:,int(M/2)]),'o-b')
f3.plot(tyfdtd,np.angle(Ezfdtd[:,int(len(Ezfdtd)/2)]),'-k')
f3.set_xlabel('x')
f3.set_ylabel(r'angle($E_{z}$)')

plt.show()

