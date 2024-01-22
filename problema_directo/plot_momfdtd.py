
#from forward import * # libreria creada
#import time as tm
#import os
import numpy as np
#from numpy import genfromtxt
import matplotlib.pyplot as plt

from forward import *

#npzfile1 = np.load('test_MoM_M_FDTD40.npz')
#M = 40

#Etmom = npzfile1['Etmom']
#Ezfdtd = npzfile1['Ezfdtd']

trans = TRANSMISOR_parameters()
trans.f = 400e6
trans.rhoS = 6.0
trans.S = 16
trans.amp = 3.8e3

acoplante = ACOPLANTE_parameters()
acoplante.f = 400e6
acoplante.epsr = 1.0  #frecuencia 1 GHz (por defecto).
acoplante.sigma = 0.0


cilindro = SCATTERER_parameters()
cilindro.epsr = 1.2 #permitividad relativa. Entre [10.0, 80.0]
cilindro.sigma = 0.0 #conductividad. Entre [0.40, 1.60]
cilindro.f = 400e6 #frecuencia 1 GHz (por defecto).
cilindro.radio = 1.0
cilindro.xc = 0.00
cilindro.yc = 0.00


receptor = RECEPTOR_parameters()
receptor.f = 400e6  #frecuencia 1 GHz (por defecto).
receptor.rhoS = 3.0 #*c/f #radio de transmisores
receptor.S = 16 #cantidad de transmisores (fuentes)

M2 = 20
tx = 5
Et,epsilon_r = RunMoM(cilindro, acoplante,trans,receptor,size_doi = 2,Tx = tx ,RES = M2)

sx = 14#15
sy = 14#15
box = [sx,sy]
a = 0.1#0.1 #meep unit
resolucion = 5 #resolución FDTD

Ezfdtd,eps_data = RunMeep(cilindro,acoplante,trans, tx, box,RES = resolucion,calibration = False, unit = a)

size_DOI = 2
#d = size_DOI/M
d2 = size_DOI/M2
#tx = d*np.linspace(-(M-1)/2,(M-1)/2,num=M,endpoint = True)
tx2 = d2*np.linspace(-(M2-1)/2,(M2-1)/2,num=M2,endpoint = True)
txfdtd = np.linspace(-7,7,num=len(Ezfdtd),endpoint = True)

fig3 = plt.figure(5)
f3 = fig3.add_subplot(221)
#f3.plot(tx,np.abs(Etmom[:,int(M/2)]),'o-b')
f3.plot(tx2,np.abs(Et[:,int(M2/2)]),'x-r')
f3.plot(txfdtd,np.abs(Ezfdtd[:,int(len(Ezfdtd)/2)]),'-k')
f3.set_xlabel('x')
f3.set_ylabel(r'abs($E_{z}$)')
f3.set_xlim([-2,2])
#f3.set_ylim([0,150])
f3 = fig3.add_subplot(222)
#f3.plot(tx,np.angle(Etmom[:,int(M/2)]),'o-b')
f3.plot(tx2,np.angle(Et[:,int(M2/2)]),'x-r')
f3.plot(txfdtd,-np.angle(Ezfdtd[:,int(len(Ezfdtd)/2)]),'-k')
f3.set_xlabel('x')
f3.set_ylabel(r'angle($E_{z}$)')
f3.set_xlim([-2,2])
f3 = fig3.add_subplot(223)
#f3.plot(tx,np.abs(Etmom[int(M/2),:]),'o-b')
f3.plot(txfdtd,np.abs(Ezfdtd[int(len(Ezfdtd)/2),:]),'-k')
f3.plot(tx2,np.abs(Et[int(M2/2),:]),'o-r')

f3.set_xlabel('y')
f3.set_ylabel(r'abs($E_{z}$)')
f3.set_xlim([-2,2])
#f3.set_ylim([0,100])
f3 = fig3.add_subplot(224)
#f3.plot(tx,np.angle(Etmom[int(M/2),:]),'o-b')
f3.plot(txfdtd,-np.angle(Ezfdtd[int(len(Ezfdtd)/2),:]),'-k')
f3.plot(tx2,np.angle(Et[int(M2/2),:]),'o-r')
f3.set_xlabel('y')
f3.set_ylabel(r'angle($E_{z}$)')
f3.set_xlim([-2,2])


plt.show()
