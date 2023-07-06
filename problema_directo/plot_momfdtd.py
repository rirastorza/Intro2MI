
#from forward import * # libreria creada
#import time as tm
#import os
import numpy as np
#from numpy import genfromtxt
import matplotlib.pyplot as plt

npzfile1 = np.load('test_MoM_M_FDTD20.npz')
M = 20

Etmom = npzfile1['Etmom']
Ezfdtd = npzfile1['Ezfdtd']

size_DOI = 2
d = size_DOI/M 
tx = d*np.linspace(-(M-1)/2,(M-1)/2,num=M,endpoint = True)
txfdtd = np.linspace(-7,7,num=len(Ezfdtd),endpoint = True)

fig3 = plt.figure(5)
f3 = fig3.add_subplot(211)
f3.plot(tx,np.abs(Etmom[:,int(M/2)]),'o-b')
f3.plot(txfdtd,np.abs(Ezfdtd[:,int(len(Ezfdtd)/2)]),'-k')
f3.set_xlabel('x')
f3.set_ylabel(r'abs($E_{z}$)')
f3 = fig3.add_subplot(212)
f3.plot(tx,np.angle(Etmom[:,int(M/2)]),'o-b')
f3.plot(txfdtd,-np.angle(Ezfdtd[:,int(len(Ezfdtd)/2)]),'-k')
f3.set_xlabel('x')
f3.set_ylabel(r'angle($E_{z}$)')

plt.show()
