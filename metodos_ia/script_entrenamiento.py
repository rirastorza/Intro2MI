#!/usr/bin/env python
# coding: utf-8

# # Uso del módulo fordward para simulación de datos de entrenamiento de red
# 
# En el módulo *forward* se implementan algunas formas de resolución del problema directo de imágenes por microondas de un arreglo circular de antenas. Para el caso analítico solo se considera el caso en que el dispersor es un cilindro centrado. En los casos del método de diferencias finitas (FDTD), elementos finitos (FEM), y el método de los momentos (MoM) se pueden generar geometrías más complejas.
# 
# - FDTD está implementado en el software [meep](https://meep.readthedocs.io/en/latest/).
# 
# - FEM está implementado en el software [FEniCS](https://fenicsproject.org/)***(IMPLEMENTAR).
# 
# - MoM está implementado completo basado en [1]***(IMPLEMENTAR).
# 
# Libros y publicaciones:
# 
# [1] Xudong Chen, Computational Methods for Electromagnetic Inverse Scattering
# 
# [2] Matteo Pastorino, Microwave Imaging
# 
# 
# Módulo Python: forward
# 
# Autores: Ramiro Irastorza 
# 
# Email: rirastorza@iflysib.unlp.edu.ar
# 

# 
# ## Ejemplo de uso para validación con modelo teórico analítico


from forward import *
import time as tm
import os
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

start_time = tm.strftime('%H:%M:%S')
f = 580e6
print('frecuencia de medición (GHz): ',f/1e6)
sx = 0.35
sy = 0.35
box = [sx,sy]
TRANSMISOR_parameters = TRANSMISOR_parameters()
TRANSMISOR_parameters.f = f
TRANSMISOR_parameters.amp = 4e4
TRANSMISOR_parameters.rhoS = 0.115
TRANSMISOR_parameters.S = 8
#Coordenadas antenas
angulo = N.linspace(0.0, 2.0*pi, 9)
xantenas = (TRANSMISOR_parameters.rhoS)*N.cos(angulo)
yantenas = (TRANSMISOR_parameters.rhoS)*N.sin(angulo)
#Generación de modelos
r = 3.34e-2/2#Nylon
#r = 2.5e-2/2#Teflon
#r = 8e-2/2#Glicerol80-Agua20
Xc = 0.0
Yc = 0.0
print('Xc:', Xc,'Yc:', Yc,'r:',r)
#Dibujo la geometría generada
cilindro = plt.Circle((Xc,Yc),r,fill = False)
frel = 20e9
sigmadc = 7e-3
epsc = 4.8+(77.0-4.8)/(1+1j*TRANSMISOR_parameters.f/frel)+sigmadc/(1j*eps0*2*pi*TRANSMISOR_parameters.f)
ACOPLANTE_parameters = ACOPLANTE_parameters()
ACOPLANTE_parameters.f = TRANSMISOR_parameters.f
ACOPLANTE_parameters.epsr = epsc.real  #frecuencia 1 GHz (por defecto).
ACOPLANTE_parameters.sigma = -epsc.imag*(eps0*2*pi*TRANSMISOR_parameters.f)#conductividad. Entre [0.40, 1.60]
#Comienzo de simulación
cilindro1 = SCATTERER_parameters()
cilindro1.epsr = 3.5 #Nylon, permitividad relativa. Entre [10.0, 80.0]
#cilindro1.epsr = 2.1 #Teflon, permitividad relativa. Entre [10.0, 80.0]
#cilindro1.epsr = 38 #Glicerol80-Agua20 a T = 20°C y f = 580MHz, permitividad relativa. Entre [10.0, 80.0]
cilindro1.sigma = 0.0
#cilindro1.sigma = 0.7#Glicerol80-Agua20 a T = 20°C y f = 580MHz, permitividad relativa. Entre [10.0, 80.0]
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
tx = 0
xS = (TRANSMISOR_parameters.rhoS)*N.cos(tx*2*pi/TRANSMISOR_parameters.S) #Coordenada x antena emisora
yS = (TRANSMISOR_parameters.rhoS)*N.sin(tx*2*pi/TRANSMISOR_parameters.S)
Ezfdtd,eps_data = RunMeep(cilindro1,ACOPLANTE_parameters,TRANSMISOR_parameters, tx, box,RES = resolucion,calibration = False)

plt.figure()
#extent2=[-0.25/2,0.25/2,-0.25/2,0.25/2]
NN = len(eps_data)
deltaX = sx/(NN)
xSint = int(((TRANSMISOR_parameters.rhoS/deltaX)*N.cos(tx*2*pi/TRANSMISOR_parameters.S)))+int(NN/2) 
ySint = int(((TRANSMISOR_parameters.rhoS/deltaX)*N.sin(tx*2*pi/TRANSMISOR_parameters.S)))+int(NN/2)
print(xSint)
print(ySint)
plt.imshow(abs(Ezfdtd).transpose(),origin='lower')#,extent = extent2,origin='lower')#cmap = 'binary')
plt.plot(xSint,ySint,'ow')
#plt.plot(xS,yS,'ow')
plt.colorbar()
#Dibujo el mapa de permitividad
NN = len(eps_data)
deltaX = sx/(NN)



x,Eztheory1 = EZ_CILINDER_LINESOURCE_MATRIZ(eps_data,cilindro1,ACOPLANTE_parameters,TRANSMISOR_parameters,tx,deltaX)


x = np.linspace(-len(eps_data)*deltaX/2., len(eps_data)*deltaX/2., len(eps_data))

nmin = np.argmin(abs(x+TRANSMISOR_parameters.rhoS))

fig2 = plt.figure(2)
f2 = fig2.add_subplot(211)
f2.plot(x,abs(Eztheory1[:,int(len(Eztheory1)/2)]),'b')
f2.plot(x,abs(Ezfdtd[:,int(len(Ezfdtd)/2)]),'.g')
f2.plot(x[nmin],0,'Xr')

f2.set_xlabel(r'y')
f2.set_ylabel(r'abs($E_{z}$)')
f2 = fig2.add_subplot(212)
f2.plot(x,N.angle(Eztheory1)[:,int(len(Eztheory1)/2)],'b')
f2.plot(x,-N.angle(Ezfdtd)[:,int(len(Ezfdtd)/2)],'.g')
f2.set_xlabel(r'y')
f2.set_ylabel(r'angle($E_{z}$)')

plt.show()

error_mod = 100*(abs(Eztheory1[nmin,int(len(Eztheory1)/2)])-abs(Ezfdtd[nmin,int(len(Ezfdtd)/2)]))/abs(Eztheory1[nmin,int(len(Eztheory1)/2)])
error_angle = 100*(N.angle(Eztheory1)[nmin,int(len(Eztheory1)/2)]+N.angle(Ezfdtd)[nmin,int(len(Ezfdtd)/2)])/N.angle(Eztheory1)[nmin,int(len(Eztheory1)/2)]

print('Error de modulo: ',error_mod,' %')
print('Error de fase: ',error_angle,' %')

print('start time: ', start_time)
print('end time:   ', tm.strftime('%H:%M:%S'))

