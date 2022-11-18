#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from forward import *
import time as tm
import os
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

#path = 'Data'
#os.mkdir(path)
start_time = tm.strftime('%H:%M:%S')

f = 1.1e9
print('frecuencia de medición (GHz): ',f/1e9)
#Fin datos experimentales

#Graficos
sx = 0.25
sy = 0.25

box = [sx,sy]

TRANSMISOR_parameters = TRANSMISOR_parameters()
TRANSMISOR_parameters.f = f
TRANSMISOR_parameters.amp = 7.5e4
TRANSMISOR_parameters.rhoS = 0.075
TRANSMISOR_parameters.S = 4.

#Coordenadas antenas
angulo = N.linspace(0.0, 2.0*pi, 5)
xantenas = (TRANSMISOR_parameters.rhoS)*N.cos(angulo)
yantenas = (TRANSMISOR_parameters.rhoS)*N.sin(angulo)

#Generación de modelos
r = 25.0e-3/2
Xc = 0.0
Yc = 0.0

print('Xc:', Xc,'Yc:', Yc,'r:',r)

#Dibujo la geometría generada
cilindro = plt.Circle((Xc,Yc),r,fill = False)
frel = 20e9
sigmadc = 400e-3
epsc = 4.8+(77.0-4.8)/(1+1j*TRANSMISOR_parameters.f/frel)+sigmadc/(1j*eps0*2*pi*TRANSMISOR_parameters.f)


ACOPLANTE_parameters = ACOPLANTE_parameters()
ACOPLANTE_parameters.f = TRANSMISOR_parameters.f
ACOPLANTE_parameters.epsr = epsc.real  #frecuencia 1 GHz (por defecto).
ACOPLANTE_parameters.sigma = -epsc.imag*(eps0*2*pi*TRANSMISOR_parameters.f)#conductividad. Entre [0.40, 1.60]


#Comienzo de simulación
cilindro1 = SCATTERER_parameters()
cilindro1.epsr = 2.1 #permitividad relativa. Entre [10.0, 80.0]
cilindro1.sigma = 0.0
cilindro1.f = TRANSMISOR_parameters.f #frecuencia 1 GHz (por defecto).
cilindro1.radio = r
cilindro1.xc = Xc
cilindro1.yc = Yc

print('Permitividad medio:',ACOPLANTE_parameters.epsr)
print('Conductividad medio:',ACOPLANTE_parameters.sigma)

print('Permitividad del cilindro:',cilindro1.epsr)
print('Conductividad del cilindro:',cilindro1.sigma)

resolucion = 5
n = resolucion*sx/a
tx = 2
xS = (0.15/2)*N.cos(tx*2*pi/4.) #Coordenada x antena emisora
yS = (0.15/2)*N.sin(tx*2*pi/4.)

Ezfdtd,eps_data = RunMeep(cilindro1,ACOPLANTE_parameters,TRANSMISOR_parameters, tx, box,RES = resolucion,calibration = False)

plt.figure()
extent2=[-0.25/2,0.25/2,-0.25/2,0.25/2]
plt.imshow(abs(Ezfdtd).transpose(),extent = extent2)#cmap = 'binary')
plt.plot(xS,yS,'ow')
plt.colorbar()
#Dibujo el mapa de permitividad
NN = len(eps_data)
deltaX = sx/(NN)

##------------------
##Solución analítica
#
x,Eztheory1 = EZ_CILINDER_LINESOURCE_MATRIZ(eps_data,cilindro1,ACOPLANTE_parameters,TRANSMISOR_parameters,tx,deltaX)

x = np.linspace(-len(eps_data)*deltaX/2., len(eps_data)*deltaX/2., len(eps_data))

fig2 = plt.figure(2)
f2 = fig2.add_subplot(211)
f2.plot(x,abs(Eztheory1[:,int(len(Eztheory1)/2)]),'b')
f2.plot(x,abs(Ezfdtd[:,int(len(Ezfdtd)/2)]),'.g')
f2.set_xlabel(r'y')
f2.set_ylabel(r'abs($E_{z}$)')
f2 = fig2.add_subplot(212)
f2.plot(x,N.angle(Eztheory1)[:,int(len(Eztheory1)/2)],'b')
f2.plot(x,-N.angle(Ezfdtd)[:,int(len(Ezfdtd)/2)],'.g')
f2.set_xlabel(r'y')
f2.set_ylabel(r'angle($E_{z}$)')


plt.show()




print('start time: ', start_time)
print('end time:   ', tm.strftime('%H:%M:%S'))
