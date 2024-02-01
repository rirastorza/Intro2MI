#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Prueba inverso Born

import numpy as np
import matplotlib.pyplot as plt
from forward import *


def InvMoM(cilindro, acoplante,trans,receptor,size_doi = 2,Tx = 1 ,RES = 40):
    '''
    Función que resuelve el problema directo por el método de los momentos.
    Se programó según el trabajo de Richmond.
    '''
    freq = trans.f #freq = 400e6
    landa = c/freq # the wavelength in the air
    k0 = 2*pi/landa # the wave number in the air
    imp = 120*pi # impedance of air
    size_DOI = size_doi # size of DOI

    Ni = trans.S # number of incidence
    Ns = receptor.S # number of receiving antennas
    theta = N.linspace(0,2*pi-2*pi/Ni, num=Ni, endpoint=True) # angle of incidence
    phi = N.linspace(0,2*pi-2*pi/Ns,num=Ns, endpoint=True) # 1 x Ns | angle of receiving antennas
    R_obs = receptor.rhoS # radius of the circle formed by receiving antennas
    Xr = R_obs*N.cos(phi) # 1 x Ns % x coordinates of receiving antennas
    Yr = -R_obs*N.sin(phi) # 1 x Ns % y coordinates of receiving antennas
    R_trans = trans.rhoS # radius of the circle formed by receiving antennas
    XT = R_trans*N.cos(theta)[Tx] # 1 x Ns % x coordinates of receiving antennas
    YT = -R_trans*N.sin(theta)[Tx] # 1 x Ns % y coordinates of receiving antennas
    epsono_r_c = cilindro.epsr # the constant relative permittivity of the object

    #Positions of the cells
    M = RES # the square containing the object has a dimension of MxM
    d = size_DOI/M #the nearest distance of two cell centers
    #print('landa: ',landa/(epsono_r_c)**.5)
    #print('d: ',d)

    tx = d*N.linspace(-(M-1)/2,(M-1)/2,num=M,endpoint = True) #((-(M-1)/2):1:((M-1)/2))*d # 1 x M
    ty = d*N.linspace((M-1)/2,-(M-1)/2,num=M,endpoint = True) #((-(M-1)/2):1:((M-1)/2))*d # 1 x M
    x, y = N.meshgrid(tx, ty)# M x M
    celldia = 2*N.sqrt(d**2/pi) # diameter of cells
    cellrad = celldia/2 #radius of cells

    #Relative permittivity of each cell
    epsono_r = N.ones((M,M))
    epsono_r[(x-cilindro.xc)**2+(y-cilindro.yc)**2 <= cilindro.radio**2] = epsono_r_c

    #print(x)
    #print(epsono_r)
    cte = (1j/2)*(pi*k0*cellrad*special.hankel2(1,k0*cellrad)-2j)#cuando estoy en la misma celda

    matrizCtes = N.zeros((len(Xr),M**2),dtype = complex)

    Xv = x.reshape((M**2,1))
    Yv = y.reshape((M**2,1))
    ji = (epsono_r.reshape((M**2,1))-1)*eps0
    tau = 1j*2*pi*freq*ji #object function or scattering potential

    #Incident wave (linea de corriente)
    absrho = ((x.T.flatten()-XT)**2+(y.T.flatten()-YT)**2)**0.5
    E_inc_roi = (-2*pi*freq*mu0/4*special.hankel2(0,k0*absrho)).T.reshape((M**2,1))

    for mm in range(len(Xr)):
        for nn in range(len(matrizCtes)):
            #if mm == nn:
                #matrizCtes[mm,nn] = (ji[nn])*cte
                #H[mm,nn] = cte
            #else:
            R = N.sqrt((Xr[mm]-Xv[nn])**2+(Yr[mm]-Yv[nn])**2)
            hmn = (1j/2)*pi*k0*cellrad*special.jv(1,k0*cellrad)*special.hankel2(0,k0*R)
            matrizCtes[mm,nn] = hmn
            #H[mm,nn] = (1j/2)*pi*k0*cellrad*special.jv(1,k0*cellrad)*special.hankel2(0,k0*R)

    N.set_printoptions(precision=3)

    print('tau: ',tau.shape)
    print('Matriz: ',matrizCtes.shape)
    T = N.diag(tau.flatten())
    print('T: ',T.shape)
    #A = [B]*tau = e^s
    A = N.matmul(N.matmul(N.matrix(matrizCtes),T),E_inc_roi)
    Es = A.flatten()
    print('Es: ',Es.shape)

    #Incident wave (linea de corriente)
    absrho = ((Xr-XT)**2+(Yr-YT)**2)**0.5
    E_inc = -2*pi*freq*mu0/4*special.hankel2(0,k0*absrho).reshape(1,len(Xr))
    Et = Es#E_inc#+
    print('Et: ',Et.shape)
    #b = N.matrix(E_inc).T
    #Solución de la ecuación lineal
    #Et = N.linalg.solve(A, b)
    #u, s, vh = N.linalg.svd(A, full_matrices=True)

    #Et = N.linalg.pinv(A,rcond=0.1)*b

    #print('tamanio A: ',Es.shape)
    #print('tamanio b: ',b.shape)

    #Et = Es+E_inc

    return Et.reshape((len(Xr),1)),epsono_r


trans = TRANSMISOR_parameters()
trans.f = 400e6

freq = trans.f #freq = 400e6
landa = c/freq # the wavelength in the air
k0 = 2*pi/landa # the wave number in the air
trans.rhoS = 6.0
trans.S = 16
trans.amp = 3.8e3

acoplante = ACOPLANTE_parameters()
acoplante.f = 400e6
acoplante.epsr = 1.0  #frecuencia 1 GHz (por defecto).
acoplante.sigma = 0.0

cilindro = SCATTERER_parameters()
cilindro.epsr = 1.4 #permitividad relativa. Entre [10.0, 80.0]
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
tx = 0
#Et,epsilon_r = InvMoM(cilindro, acoplante,trans,receptor,size_doi = 2,Tx = tx ,RES = M2)


Et, epsilon_r = InvMoM(cilindro, acoplante,trans,receptor,size_doi = 2,Tx = tx ,RES = 20)

print(Et)
#Calculo del campo total medido
sx = 14#15
sy = 14#15
box = [sx,sy]
a = 0.1#0.1 #meep unit
resolucion = 5 #resolución FDTD

Ezfdtd,eps_data = RunMeep(cilindro,acoplante,trans, tx, box,RES = resolucion,calibration = False, unit = a)

Ezfdtdinc,eps_data = RunMeep(cilindro,acoplante,trans, tx, box,RES = resolucion,calibration = True, unit = a)

#Ni = trans.S # number of incidence
#Ns = receptor.S # number of receiving antennas
#theta = N.linspace(0,2*pi-2*pi/Ni, num=Ni, endpoint=True) # angle of incidence
#phi = N.linspace(0,2*pi-2*pi/Ns,num=Ns, endpoint=True) # 1 x Ns | angle of receiving antennas
#R_obs = receptor.rhoS # radius of the circle formed by receiving antennas
#Xr = R_obs*N.cos(phi) # 1 x Ns % x coordinates of receiving antennas
#Yr = -R_obs*N.sin(phi) # 1 x Ns % y coordinates of receiving antennas
#R_trans = trans.rhoS # radius of the circle formed by receiving antennas
#XT = R_trans*N.cos(theta)[tx] # 1 x Ns % x coordinates of receiving antennas
#YT = -R_trans*N.sin(theta)[tx] # 1 x Ns % y coordinates of receiving antennas

##Incident wave (linea de corriente)
#absrho = ((Xr-XT)**2+(Yr-YT)**2)**0.5
#E_inc = -2*pi*freq*mu0/4*special.hankel2(0,k0*absrho).reshape(1,len(Xr))


#ESzfdtd = Ezfdtd-


Ni = trans.S # number of incidence
Ns = receptor.S # number of receiving antennas
phi = np.linspace(0,2*pi-2*pi/Ns,num=Ns, endpoint=True) # 1 x Ns | angle of receiving antennas
R_obs = receptor.rhoS # radius of the circle formed by receiving antennas
Xr = R_obs*np.cos(phi) # 1 x Ns % x coordinates of receiving antennas
Yr = -R_obs*np.sin(phi) # 1 x Ns % y coordinates of receiving antennas

Es_mfdtd = []
xint = []
yint = []
for mm in range(len(Xr)):
        xDint = int(resolucion*Xr[mm]/a)+int(len(Ezfdtd)/2) #Coordenada x antena receptora en entero
        yDint = int(resolucion*Yr[mm]/a)+int(len(Ezfdtd)/2) #Coordenada y antena receptora en entero
        xint.append(xDint)
        yint.append(yDint)
        es = Ezfdtd[xDint,yDint]-Ezfdtdinc[xDint,yDint]
        Es_mfdtd.append(es)
#print(Et[:16])

fig3 = plt.figure(5)
f3 = fig3.add_subplot(311)
#f3.plot(tx,np.abs(Etmom[:,int(M/2)]),'o-b')
#f3.plot(np.abs(Et),'x-r')
f3.plot(np.abs(Es_mfdtd),'o-b')
#f3.plot(txfdtd,np.abs(Ezfdtd[:,int(len(Ezfdtd)/2)]),'-k')
f3.set_xlabel('x')
f3.set_ylabel(r'abs($E_{z}$)')
#f3.set_xlim([-2,2])
#f3.set_ylim([0,150])
f3 = fig3.add_subplot(312)
#f3.plot(tx,np.abs(Etmom[:,int(M/2)]),'o-b')
f3.plot(np.abs(Et),'x-r')

f3 = fig3.add_subplot(313)
#f3.plot(tx,np.angle(Etmom[:,int(M/2)]),'o-b')
f3.plot(np.angle(Et),'x-r')
f3.plot(-np.angle(Es_mfdtd),'o-b')
#f3.plot(txfdtd,-np.angle(Ezfdtd[:,int(len(Ezfdtd)/2)]),'-k')
f3.set_xlabel('x')
f3.set_ylabel(r'angle($E_{z}$)')
#f3.set_xlim([-2,2])
#f3 = fig3.add_subplot(223)
##f3.plot(tx,np.abs(Etmom[int(M/2),:]),'o-b')
##f3.plot(txfdtd,np.abs(Ezfdtd[int(len(Ezfdtd)/2),:]),'-k')
#f3.plot(tx2,np.abs(Et[int(M2/2),:]),'o-r')

#f3.set_xlabel('y')
#f3.set_ylabel(r'abs($E_{z}$)')
#f3.set_xlim([-2,2])
##f3.set_ylim([0,100])
#f3 = fig3.add_subplot(224)
##f3.plot(tx,np.angle(Etmom[int(M/2),:]),'o-b')
##f3.plot(txfdtd,-np.angle(Ezfdtd[int(len(Ezfdtd)/2),:]),'-k')
#f3.plot(tx2,np.angle(Et[int(M2/2),:]),'o-r')
#f3.set_xlabel('y')
#f3.set_ylabel(r'angle($E_{z}$)')
#f3.set_xlim([-2,2])


plt.figure()
#extent2=[-0.25/2,0.25/2,-0.25/2,0.25/2]
plt.imshow(abs(Ezfdtd).transpose())#,extent = extent2)#cmap = 'binary')
plt.plot(xint,yint,'ow')
plt.colorbar()

plt.show()
