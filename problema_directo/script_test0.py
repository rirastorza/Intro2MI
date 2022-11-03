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

#Dato experimentales
path = '/home/ramiro/Documentos/CIC-2022-kira/Imágenes por microondas/Mediciones IAR/Mediciones 20-9-2022/ITM_Medida_20SEP2022/'
blanco = 'MED_prueba_inicial_Npos_4_20-Sep-2022_15h_12m_24s_pos_completo_col.txt'
concilindro = 'MED_prueba_inicial_Npos_4_20-Sep-2022_15h_28m_2s_pos_completo_col.txt'
datos = N.loadtxt(path+blanco,skiprows=1)
mfrec = 140#8 baja, 140 alta
f = datos[mfrec,0]*1e6
print('frecuencia de medición (GHz): ',f/1e9)
#Fin datos experimentales

#Graficos
sx = 0.25
sy = 0.25

box = [sx,sy]
f = 1.1e9
TRANSMISOR_parameters = TRANSMISOR_parameters()
TRANSMISOR_parameters.f = f
TRANSMISOR_parameters.amp = 7500.
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

#figure, axes = plt.subplots()
#plt.xlim(-0.25/2, 0.25/2)
#plt.ylim(-0.25/2 , 0.25/2)
#axes.set_aspect(1)
#axes.add_artist(cilindro)
#axes.plot(xantenas,yantenas,'ok')

#plt.show()

frel = 20e9

sigmadc = 400e-3
frec = N.linspace(100e6,3e9,1000)
epsc = 4.8+(77.0-4.8)/(1+1j*frec/frel)+sigmadc/(1j*eps0*2*pi*frec)

#fig2 = plt.figure(2)
#ax1 = fig2.add_subplot(121)
#ax1.semilogx(frec,epsc.real,'-k')
#ax2 = fig2.add_subplot(122)
##conductividad = -epsc.imag*(eps0*2*pi*frec)
#ax2.semilogx(frec,-epsc.imag,'-b')
#plt.show()



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
Tx = N.arange(4)
Tr = N.arange(4)
EzTr = N.zeros((4,4))
sx = 0.25
sy = 0.25
box = [sx,sy]

tx = 2
Ezfdtd,eps_data = RunMeep(cilindro1,ACOPLANTE_parameters,TRANSMISOR_parameters, tx, box,RES = 5,calibration = False)
Ezfdtdinc,eps_data_no = RunMeep(cilindro1,ACOPLANTE_parameters,TRANSMISOR_parameters, tx, box,RES = 5,calibration = True)

tr = 0
xSint = int(resolucion*((0.15/2)*N.cos(tr*2*pi/4.))/a)+int(n/2) #Coordenada x antena emisora
ySint = int(resolucion*((0.15/2)*N.sin(tr*2*pi/4.))/a)+int(n/2)
EzTr[tx,tr] = abs(Ezfdtd)[xSint,ySint]/abs(Ezfdtdinc)[xSint,ySint]

#print('Teflon: Tx: ',tx,' Tr: ',tr,'Campo: ',abs(Ezfdtd)[xSint,ySint])
#print('Incidente: Tx: ',tx,' Tr: ',tr,'Campo: ',abs(Ezfdtdinc)[xSint,ySint])

xS = (0.15/2)*N.cos(Tx*2*pi/4.) #Coordenada x antena emisora
yS = (0.15/2)*N.sin(Tx*2*pi/4.)

plt.figure()
extent2=[-0.25/2,0.25/2,-0.25/2,0.25/2]
plt.imshow(abs(Ezfdtd).transpose(),extent = extent2)#cmap = 'binary')
plt.plot(xS,yS,'ow')
plt.colorbar()
#Dibujo el mapa de permitividad
plt.figure()
plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary',extent = extent2)#.transpose()

plt.plot(xS,yS,'ow')
plt.show()

ez = N.zeros(4)
ezinc = N.zeros(4)
for tr in Tr:
    xSint = int(resolucion*((0.15/2)*N.cos(tr*2*pi/4.))/a)+int(n/2) #Coordenada x antena emisora
    ySint = int(resolucion*((0.15/2)*N.sin(tr*2*pi/4.))/a)+int(n/2)
    ez[tr] = abs(Ezfdtd)[xSint,ySint]
    ezinc[tr] = abs(Ezfdtdinc)[xSint,ySint]
    #print('Teflon: Tx: ',tx,' Tr: ',tr,'Campo: ',abs(Ezfdtd)[xSint,ySint])
    #print('Incidente: Tx: ',tx,' Tr: ',tr,'Campo: ',abs(Ezfdtdinc)[xSint,ySint])


print('ez:',ez)
print('ezinc:',ezinc)

#Comparo con mediciones

datos = N.loadtxt(path+blanco,skiprows=1)
s_juntas_inc = datos[mfrec,6]
s_op_inc = datos[mfrec,5]
datos = N.loadtxt(path+concilindro,skiprows=1)
s_op = datos[mfrec,5]
print('Antenas opuestas:')
print('sRXTXinc: ',s_op_inc)
print('ezincRX/ezincTX: ',20*N.log10(ezinc[0]/ezinc[2]))
print('Antenas juntas:')
print('sRXTXinc: ',s_juntas_inc)
print('ezincRX/ezincTX: ',20*N.log10(ezinc[1]/ezinc[2]))
print('Diferencias medidas (%): ',100*(s_op_inc-s_juntas_inc)/s_juntas_inc)
print('Diferencias simuladas (%): ',100*((10*N.log10(ezinc[0]/ezinc[2])-10*N.log10(ezinc[1]/ezinc[2]))/10*N.log10(ezinc[1]/ezinc[2])))


##-------------------
##Varias antenas
#for tx in Tx:
    #Ezfdtd,eps_data = RunMeep(cilindro1,ACOPLANTE_parameters,TRANSMISOR_parameters, tx, box,RES = 5,calibration = False)
    #Ezfdtdinc,eps_data_no = RunMeep(cilindro1,ACOPLANTE_parameters,TRANSMISOR_parameters, tx, box,RES = 5,calibration = True)
    #for tr in Tr:
        #xSint = int(resolucion*((0.15/2)*N.cos(tr*2*pi/4.))/a)+int(n/2) #Coordenada x antena emisora
        #ySint = int(resolucion*((0.15/2)*N.sin(tr*2*pi/4.))/a)+int(n/2)
        #EzTr[tx,tr] = abs(Ezfdtd)[xSint,ySint]/abs(Ezfdtdinc)[xSint,ySint]
        #print('Teflon: Tx: ',tx,' Tr: ',tr,'Campo: ',abs(Ezfdtd)[xSint,ySint])
        #print('Incidente: Tx: ',tx,' Tr: ',tr,'Campo: ',abs(Ezfdtdinc)[xSint,ySint])
    ##print('Campo en emisor:',EzTr[tx,tx])
    ##plt.figure()
    ##plt.imshow(abs(Ezfdtd),)#cmap = 'binary')
    ##plt.colorbar()
    ##plt.show()


#Dibujo la imagen de entrada

##Dibujo el mapa de permitividad
#plt.figure()
#plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
#plt.show()


print('start time: ', start_time)
print('end time:   ', tm.strftime('%H:%M:%S'))
