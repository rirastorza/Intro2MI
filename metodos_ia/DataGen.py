#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Script generador de data de calibracion para 
#tomografia de microondas. Simulaciones de cilindros
#de diverso radio y propiedades dielectricas.

from scipy import *
import numpy as N
import os
from scipy.constants import epsilon_0, pi
#import h5py
from forward import *
import time as tm
from matplotlib import pyplot as plt
from skimage.restoration import unwrap_phase
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

start_time = tm.strftime('%H:%M:%S')

#TRANSMISOR_parameters.f = 1.3e9
#TRANSMISOR_parameters.amp = 7500.
#ACOPLANTE_parameters.f = 1.3e9
#ACOPLANTE_parameters.epsr = 23.3  #frecuencia 1 GHz (por defecto).
#ACOPLANTE_parameters.sigma = 1.3
#TRANSMISOR_parameters.rhoS = 0.075

f = 580e6
print('frecuencia de medición (GHz): ',f/1e6)
sx = 0.25
sy = 0.25
box = [sx,sy]
TRANSMISOR_parameters = TRANSMISOR_parameters()
TRANSMISOR_parameters.f = f
TRANSMISOR_parameters.amp = 4e4
TRANSMISOR_parameters.rhoS = 0.075
TRANSMISOR_parameters.S = 8

resolucion = 5

frel = 20e9
sigmadc = 7e-3
epsc = 4.8+(77.0-4.8)/(1+1j*TRANSMISOR_parameters.f/frel)+sigmadc/(1j*eps0*2*pi*TRANSMISOR_parameters.f)
ACOPLANTE_parameters = ACOPLANTE_parameters()
ACOPLANTE_parameters.f = TRANSMISOR_parameters.f
ACOPLANTE_parameters.epsr = epsc.real  #frecuencia 1 GHz (por defecto).
ACOPLANTE_parameters.sigma = -epsc.imag*(eps0*2*pi*TRANSMISOR_parameters.f)#conductividad. Entre [0.40, 1.60]

n = resolucion*sx/a

def CampoUnaAntena(ACOPLANTE_parameters,TRANSMISOR_parameters,Tr=0, rad=0.015, xc=0.0, yc = 0.0, eps=80.0, sigma=1.9):
    '''
    Calcula la distribucion de campo con la antena en posicion Tr emitiendo
    '''
    cilindro1 = SCATTERER_parameters()
    cilindro1.epsr = eps
    cilindro1.sigma = sigma
    cilindro1.f = TRANSMISOR_parameters.f
    cilindro1.radio = rad
    cilindro1.xc = xc
    cilindro1.yc = yc
    radiocilindro = cilindro1.radio
    Tx = Tr #emisora
    rhoS = TRANSMISOR_parameters.rhoS #(7.5cm)
    
    xt = (rhoS)*N.cos(Tx*2*pi/TRANSMISOR_parameters.S) #Coordenada x antena transmisora
    yt = (rhoS)*N.sin(Tx*2*pi/TRANSMISOR_parameters.S) #Coordenada y antena transmisora

    sx = 0.25 #(0.25 m)
    sy = 0.25

    #calibration = False # si hay o no cilindro
    center = [xc,yc]
    box = [sx,sy]
    resolucion = 5
    print("Corrida con cilindro, Tx:",Tx)

    Ezfdtd,eps_data = RunMeep(cilindro1,ACOPLANTE_parameters,TRANSMISOR_parameters, Tx, box,RES = resolucion,calibration = False)

    #flag = generaCTL(problemname+".ctl",center,SCATTERER_parameters,ACOPLANTE_parameters,TRANSMISOR_parameters,1, Tx, box,tsim,resolucion)
    #string = "meep no-bone?=false "+problemname+".ctl"
    #os.system(string)
    
    #print("Graficando el mapa de permitividades...")
    #string = "h5topng -S3 "+problemname+"-eps-000000.00.h5"
    #os.system(string)
    
    #
    #-Fin de generación de h5
    #
    
    #filename = problemname+"-eps-000000.00.h5"
    #import h5py
    #f = h5py.File(filename, 'r')
    #print("Keys: %s" % f.keys())
    #a_group_key = f.keys()[0]
    #epsilon = list(f[a_group_key])
    #NN = len(eps_data)
    #deltaX = sx/(NN)
    
    ##------------------
    print('start time: ', start_time)
    print('end time:   ', tm.strftime('%H:%M:%S'))
    
    #Campo genereado con Meep (FDTD)
    #filename = problemname+"-EzTx"+str(Tx)+".h5"
    #f = h5py.File(filename, 'r')
    #a_group_key = f.keys()[0]
    #ezti = N.asarray(list(f[a_group_key]))
    #a_group_key = f.keys()[1]
    #eztr = N.asarray(list(f[a_group_key]))
    Ezmeep = Ezfdtd#eztr[:,:,0] +1.0j*ezti[:,:,0]
    
    #Phase unwrapping
    sx = 250.0e-3
    NN = len(eps_data)
    deltaX = sx/(NN)
    #a = 0.005 #Unidad de meep
    #print(n)
    xSint = int(deltaX*((0.15/2)*N.cos(Tx*2*pi/TRANSMISOR_parameters.S)))+int(NN/2) #Coordenada x antena emisora
    ySint = int(deltaX*((0.15/2)*N.sin(Tx*2*pi/TRANSMISOR_parameters.S)))+int(NN/2)
    print(xSint)
    print(ySint)
    print(Ezmeep.shape)
    EzTx = abs(Ezmeep)[xSint,ySint]
    print(EzTx)
    image_unwrapped = unwrap_phase(N.angle(Ezmeep))
    phaseTx = image_unwrapped[xSint,ySint]
    phaseuw = image_unwrapped-phaseTx
    EzR = Ezfdtd.real#eztr[:,:,0]
    EzIm = Ezfdtd.imag#ezti[:,:,0]
    return EzR,EzIm,phaseuw,Ezmeep

for j in range(877,1000):
    Datos = []
    semilla = int(j+100) #La semilla del generado será el número del modelo 
    N.random.seed(semilla)#Seteo la semilla del generador
    r = N.random.uniform(0.002,0.030,int(1e4))
    N.random.seed(semilla+10)
    radscat = N.random.uniform(-TRANSMISOR_parameters.rhoS-(-0.005-r),TRANSMISOR_parameters.rhoS-(0.005+r),int(1e4))
    N.random.seed(semilla+20)
    anglescat = N.random.uniform(0.0,2*pi,int(1e4))
    Xc = radscat*N.cos(anglescat)
    Yc = radscat*N.sin(anglescat)
    
    N.random.seed(semilla+30)
    epsi = N.random.uniform(2.0,80.0,int(1e4))
    N.random.seed(semilla+40)
    sig = N.random.uniform(0.0,1.60,int(1e4))
    print('Iniciando simulacion con:')
    print('radio: ',r[j])
    print('Xc: ',Xc[j])
    print('Yc: ',Yc[j])
    print('Epsilon: ',epsi[j])
    print('Sigma: ',sig[j])
    
    #Grafico cilindro
    fig, ax = plt.subplots()
    ax.add_patch(plt.Circle((0, 0), TRANSMISOR_parameters.rhoS, color='r', alpha=0.5))
    ax.add_patch(plt.Circle((Xc[j], Yc[j]), r[j], color='b', alpha=0.5))
    #Use adjustable='box-forced' to make the plot area square-shaped as well.
    ax.set_aspect('equal', adjustable='datalim')
    ax.plot()   #Causes an autoscale update.
    plt.show()
    input("Press Enter to continue...")
    
    archivo = 'Data'+str(j+1)+'.out'
    archpar = 'Param'+str(j+1)+'.out'
    f = open('Data/'+archivo, "a")
    par = open('Data/'+archpar, "a")
    f.write('#Ez Real / Ez Complejo / Modulo de Campo / Fase Unwrapped '+' \n')
    par.write('#Parametros: Radio / Xc / Yc / Epsilon / Sigma '+' \n')
    for i in range(TRANSMISOR_parameters.S):
        EzR,EzIm,phaseuw,Ezmeep = CampoUnaAntena(ACOPLANTE_parameters,TRANSMISOR_parameters,Tr=i, rad=r[j], xc=Xc[j], yc =Yc[j], eps=epsi[j], sigma=sig[j])
        #problemname = "cilindroNUEVO"
        #sx = 250.0e-3
        #filename = problemname+"-eps-000000.00.h5"
        #fil = h5py.File(filename, 'r')
        #a_group_key = fil.keys()[0]
        #epsilon = list(fil[a_group_key])
        #NN = len(epsilon)
        #deltaX = sx/(NN)
        
        for k in range(TRANSMISOR_parameters.S):
            idx = mod(i+k,TRANSMISOR_parameters.S)
            Xr = int(resolucion*((0.15/2)*N.cos(idx*2*pi/TRANSMISOR_parameters.S))/a)+int(n/2) #Coordenadas antena receptora
            Yr = int(resolucion*((0.15/2)*N.sin(idx*2*pi/TRANSMISOR_parameters.S))/a)+int(n/2)
            Datos.append([EzR[Xr,Yr],EzIm[Xr,Yr],abs(Ezmeep)[Xr,Yr],phaseuw[Xr,Yr]])
            f.write('%.15f'%Datos[TRANSMISOR_parameters.S*i+k][0]+' '+'%.15f'%Datos[TRANSMISOR_parameters.S*i+k][1]+' '+'%.15f'%Datos[TRANSMISOR_parameters.S*i+k][2]+' '+'%.15f'%Datos[TRANSMISOR_parameters.S*i+k][3]+' \n')
    par.write(str('%.15f'%r[j])+' \n')
    par.write(str('%.15f'%Xc[j])+' \n') 
    par.write(str('%.15f'%Yc[j])+' \n')
    par.write(str('%.15f'%epsi[j])+' \n')
    par.write(str('%.15f'%sig[j])+' \n')
    f.close()
    par.close()
    #string = "rm *.h5"
    #os.system(string)
    #string = "rm *.ctl"
    #os.system(string)
    #string = "rm *.png"
    #os.system(string)
