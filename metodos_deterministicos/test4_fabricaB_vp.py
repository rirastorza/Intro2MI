#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Problema inverso basado en la aproximación 
#tipo Born.
#26/9/2018

from __future__ import print_function

import os
from funciones_canonicos import *
import time as tm
from dolfin import *
from matplotlib import pyplot as plt

#start_time = tm.strftime('%H:%M:%S')

SCATTERER_parameters.epsr = 1.2
SCATTERER_parameters.sigma = 0.0
SCATTERER_parameters.f = 1.0e9
SCATTERER_parameters.radio = 0.5*c/SCATTERER_parameters.f
TRANSMISOR_parameters.f = 1.0e9
TRANSMISOR_parameters.amp = 7500.
ACOPLANTE_parameters.f = 1.0e9
ACOPLANTE_parameters.epsr = 1.0  #frecuencia 1 GHz (por defecto).
ACOPLANTE_parameters.sigma = 0.0

resolu = 5


for Tx in range(int(TRANSMISOR_parameters.S)):
    print("Corrida sin hueso, Tx:",Tx)

    rhoS = 1.5*c/TRANSMISOR_parameters.f
    xt = (rhoS)*N.cos(Tx*2*pi/TRANSMISOR_parameters.S) #Coordenada x antena transmisora
    yt = (rhoS)*N.sin(Tx*2*pi/TRANSMISOR_parameters.S) #Coordenada y antena transmisora
    phi_s, rho_s  = cart2pol(xt, yt)


    #Receptores (los M = 55 opuestos al transmisor según Pastorino en un arco de 270 °C)
    M = 25
    MM = N.arange(1,M+1)
    xm = []
    ym = []
    for m in MM:
        phi_i = Tx*2*pi/TRANSMISOR_parameters.S
        phi_m = phi_i +pi/4.+(m-1)*3.*pi/(2.*(M-1))
        xm.append((rhoS)*N.cos(phi_m)) #Coordenada x antena transmisora
        ym.append((rhoS)*N.sin(phi_m)) #Coordenada y antena transmisora


    problemname = "elipse_inverso_des1.2"
    landa = c/TRANSMISOR_parameters.f
    sx = 4.0*landa #(aproximadamente 1.2 m)
    sy = 4.0*landa

    filename = problemname+"-eps-000000.00.h5"
    import h5py
    f = h5py.File(filename, 'r')
    #print("Keys: %s" % f.keys())
    a_group_key = f.keys()[0]
    epsilon = list(f[a_group_key])#Ojo! epsilon es una lista!
    NN = len(epsilon)
    deltaX = sx/(NN)

    epsilon = N.asarray(epsilon)


    extent=[-len(epsilon[0,:])*deltaX/2./landa,len(epsilon[0,:])*deltaX/2./landa,-len(epsilon[0,:])*deltaX/2./landa,len(epsilon[0,:])*deltaX/2./landa]

    fig, axes = plt.subplots(4, 1, figsize=(8, 8),subplot_kw={'adjustable': 'box-forced'})#, sharex=True, sharey=True
    #ax = axes.ravel()
    ax = axes

    ax[0].imshow(epsilon, cmap=plt.cm.gray, interpolation='nearest',extent = extent,origin = 'lower')#
    ax[0].plot(N.asarray(ym)/landa,N.asarray(xm)/landa,'ow')
    ax[0].plot(N.asarray(yt)/landa,N.asarray(xt)/landa,'xw')



    #plt.savefig('foo.png')


    f = h5py.File(filename, 'r')
    print("Keys: %s" % f.keys())
    a_group_key = f.keys()[0]
    epsilon = list(f[a_group_key])
    NN = len(epsilon)
    deltaX = sx/(NN)
    x = N.linspace(-len(epsilon)*deltaX/2., len(epsilon)*deltaX/2., len(epsilon))


    x_10,Ezmeep_10, phase_10 = campoEmeep(TRANSMISOR_parameters,problemname,sx = sx, Tx = Tx, resolu = resolu,)
    xinc_10,Ezmeepinc_10, phaseinc_10 = campoEmeepinc(TRANSMISOR_parameters,problemname,sx = sx,Tx = Tx, resolu = resolu,)


    Es = Ezmeep_10*N.exp(-1j*phase_10)-Ezmeepinc_10*N.exp(-1j*phaseinc_10)



    #Partición del dominio de búsqueda

    #Primero selecciono la zona de búqueda
    a = 0.05
    NN = resolu*sx/a
    Nmin = -int(resolu*landa/a)+int(NN/2)
    Nmax = int(resolu*landa/a)+int(NN/2)
    print(Nmax-Nmin,type(Nmin),type(epsilon))
    extent2=[-landa/landa,landa/landa,-landa/landa,landa/landa]

    ax[1].imshow(N.asarray(epsilon)[Nmin:Nmax,Nmin:Nmax], cmap=plt.cm.gray, interpolation='nearest',origin = 'lower',extent = extent2)

    #N = 29*29#En este caso lo parto en 29, y un tamaño de 2*lambda
    from skimage.transform import rescale, resize, downscale_local_mean
    epsilonBusqueda = N.asarray(epsilon)[Nmin:Nmax,Nmin:Nmax]
    
    zonaBusqueda = resize(epsilonBusqueda,(epsilonBusqueda.shape[0]/2, epsilonBusqueda.shape[1]/2))

    #ax[2].imshow(zonaBusqueda, cmap=plt.cm.gray, interpolation='nearest',origin = 'lower',extent = extent2)

    #Defino el vector Tau (pag 18 de libro de Pastorino)
    omega =2.*pi*TRANSMISOR_parameters.f
    tau = (1.0j)*omega*eps0*(N.reshape(zonaBusqueda, len(zonaBusqueda)**2)-1)#función objeto o scattering potential, en este caso es solo imaginaria 

    #Coordenadas de la zona de búsqueda
    deltaXbusqueda = 2.*landa/len(zonaBusqueda)
    xn = N.linspace(-len(zonaBusqueda)*deltaXbusqueda/2.+deltaXbusqueda/2, len(zonaBusqueda)*deltaXbusqueda/2.-deltaXbusqueda/2, len(zonaBusqueda))
    yn = N.linspace(-len(zonaBusqueda)*deltaXbusqueda/2.+deltaXbusqueda/2, len(zonaBusqueda)*deltaXbusqueda/2.-deltaXbusqueda/2, len(zonaBusqueda))
    xngrid, yngrid = N.meshgrid(xn, yn)

    xnv = N.reshape(xngrid,len(zonaBusqueda)**2)
    ynv = N.reshape(yngrid,len(zonaBusqueda)**2)

    #ax[2].plot(xnv/landa,ynv/landa,'or')


    #Ahora construyo la ecuación 5.12.4 de Pastorino
    # [B]tau = es
    
    Bmatriz = N.zeros((M,len(zonaBusqueda)**2),dtype = complex)


    def funcr(x,y,xm,ym,xt,yt,kb,acoplante):
        absrho1 = ((x-xm)**2.0+(y-ym)**2.0)**0.5
        Ezinc = EINC_LINESOURCE([xt,yt], [x,y],acoplante)
        return (Ezinc*(1j/4.0)*special.hankel2(0, kb*absrho1)).real

    def funci(x,y,xm,ym,xt,yt,kb,acoplante):
        absrho1 = ((x-xm)**2.0+(y-ym)**2.0)**0.5
        Ezinc = EINC_LINESOURCE([xt,yt], [x,y],acoplante)
        return (Ezinc*(1j/4.0)*special.hankel2(0, kb*absrho1)).imag

    Ym = N.asarray(ym)
    Xm = N.asarray(xm)
    kb = omega*((mu0*eps0)**0.5) #numero de onda

    from scipy import integrate

    for mm in N.arange(M):
        for nn in N.arange(len(zonaBusqueda)**2):
            args = Xm[mm], Ym[mm],xt,yt,kb,ACOPLANTE_parameters
            g = lambda x: ynv[nn]-deltaXbusqueda/2.
            h = lambda x: ynv[nn]+deltaXbusqueda/2.
            integral_real = integrate.dblquad(funcr, xnv[nn]-deltaXbusqueda/2., xnv[nn]+deltaXbusqueda/2.,g,h, args= args)
            integral_imag = integrate.dblquad(funci, xnv[nn]-deltaXbusqueda/2., xnv[nn]+deltaXbusqueda/2.,g,h, args= args)

            integral = integral_real[0]+1j*integral_imag[0]
            Bmatriz[mm,nn] = (1.0j)*omega*mu0*integral
            #Bmatriz_imag[mm,nn] = (1.0j)*omega*mu0
            
        

    BB = N.matrix(Bmatriz)
    TAU = N.matrix(tau).T

    Es_estimado = BB*TAU

    #Comparo con el simulado
    Es_m = []
    for mm in N.arange(M):
        xDint = int(resolu*Xm[mm]/a)+int(len(Es)/2) #Coordenada x antena receptora en entero
        yDint = int(resolu*Ym[mm]/a)+int(len(Es)/2) #Coordenada y antena receptora en entero
        Es_m.append(Es[xDint,yDint])

    ax[2].plot(abs(Es_estimado),'or')
    ax[2].plot(abs(N.asarray(Es_m)),'-b')

    ax[3].plot(N.angle(Es_estimado),'or')
    ax[3].plot(N.angle(N.asarray(Es_m)),'-b')
    
    #Guardo la matriz
    filenombre = 'B_Tx'+str(Tx)
    N.save(filenombre, BB)

    plt.savefig('figura_'+str(Tx)+'.png')
