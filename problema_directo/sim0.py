#!/usr/bin/env python
# coding: utf-8

from forward import *
import time as tm
import os
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt


def agrupa(datos,nfrec,nant):
    """
    Agrupa los datos medidos de la siguiente forma (para nant = 8):
    #Med_P1P2  Med_P1P3  Med_P1P4  Med_P1P5  Med_P1P6  Med_P1P7  Med_P1P8  
    #Med_P2P3  Med_P2P4  Med_P2P5  Med_P2P6  Med_P2P7  Med_P2P8  Med_P1P2
    #Med_P3P4  Med_P3P5  Med_P3P6  Med_P3P7  Med_P3P8  Med_P1P3  Med_P2P3  
    #Med_P4P5  Med_P4P6  Med_P4P7  Med_P4P8  Med_P1P4  Med_P2P4  Med_P3P4
    #Med_P5P6  Med_P5P7  Med_P5P8  Med_P1P5  Med_P2P5  Med_P3P5  Med_P4P5
    #Med_P6P7  Med_P6P8  Med_P1P6  Med_P2P6  Med_P3P6  Med_P4P6  Med_P5P6
    #Med_P7P8  Med_P1P7  Med_P2P7  Med_P3P7  Med_P4P7  Med_P5P7  Med_P6P7
    #Med_P1P8  Med_P2P8  Med_P3P8  Med_P4P8  Med_P5P8  Med_P6P8  Med_P7P8
    """
    Sinci = np.zeros((len(datos[0,1:nant]),len(datos[0,1:nant])))
    
    nant_anterior = 1
    for i in range(nant-1):
        #print(i,nant-i-1,nant_anterior,nant_anterior+nant-i)
        Sinci[i,:nant-i-1] = datos[nfrec,nant_anterior:nant_anterior+nant-i-1] 
        nant_anterior = nant_anterior+nant-i-1
    #print(Sinci)
    for i in range(nant-1):
        rows, cols = np.indices((len(datos[0,1:nant]),len(datos[0,1:nant])))
        row_vals = np.diag(np.fliplr(rows), k=-i-1)
        col_vals = np.diag(np.fliplr(cols), k=-i-1)
        Sinci[row_vals, col_vals]= Sinci[i,0:nant-2-i]
        #print(Sinci[i,0:nant-1-i])
    
    return Sinci[:,:]




incidente = 'MED_NaCL_8pos_sin_teflon_Npos_8_04-Apr-2023_15h_11m_30s_pos_completo_col.txt'
canonico = 'MED_NaCL_8pos_teflon_Npos_8_04-Apr-2023_14h_44m_26s_pos_completo_col.txt'

datos = np.loadtxt(incidente,skiprows=1)
f_medida = 1e6*datos[:,0]
datos_canonico = np.loadtxt(canonico,skiprows=1,delimiter=',')

nfrec = 64
nant = 8
Sinc_FINAL = agrupa(datos,nfrec,nant)
S_FINAL = agrupa(datos_canonico,nfrec,nant)

f = f_medida[nfrec]

start_time = tm.strftime('%H:%M:%S')

#f = 0.6e9
print('frecuencia de medición (GHz): ',f/1e9)
sx = 0.3
sy = 0.3
box = [sx,sy]

TRANSMISOR_parameters = TRANSMISOR_parameters()
TRANSMISOR_parameters.f = f
TRANSMISOR_parameters.amp = 7.5e4
TRANSMISOR_parameters.rhoS = 0.075
TRANSMISOR_parameters.S = 8.

#Debye para graficar
frel = 20e9
sigmadc = 8e-3*100
frec = N.linspace(100e6,3e9,1000)#para dibujar
epsc = 4.8+(76.0-4.8)/(1+1j*frec/frel)+sigmadc/(1j*eps0*2*pi*frec)#para dibujar

fig2 = plt.figure(1)
ax1 = fig2.add_subplot(121)
ax1.semilogx(frec,epsc.real,'-k')
ax1.set_xlim([0.6e9, 1.6e9])
ax1.set_ylabel('Parte real permitividad')
ax1.set_xlabel('Frecuencia (Hz)')
ax2 = fig2.add_subplot(122)
ax2.semilogx(frec,-epsc.imag,'-b')
ax2.set_xlim([0.6e9, 1.6e9])
ax2.set_ylabel('Parte imaginaria permitividad')
ax2.set_xlabel('Frecuencia (Hz)')

#Definición de Debye para simular:
epsc = 4.8+(76.0-4.8)/(1+1j*TRANSMISOR_parameters.f/frel)+sigmadc/(1j*eps0*2*pi*TRANSMISOR_parameters.f)
ACOPLANTE_parameters = ACOPLANTE_parameters()
ACOPLANTE_parameters.f = TRANSMISOR_parameters.f
ACOPLANTE_parameters.epsr = epsc.real 
ACOPLANTE_parameters.sigma = -epsc.imag*(eps0*2*pi*TRANSMISOR_parameters.f)
#plt.show()

#Coordenadas antenas
angulo = N.linspace(0.0, 2.0*pi, 9)
xantenas = (TRANSMISOR_parameters.rhoS)*N.cos(angulo)
yantenas = (TRANSMISOR_parameters.rhoS)*N.sin(angulo)

#Generación de modelos
r = 25.0e-3/2.0
Xc = 0.0
Yc = 0.0

print('Xc:', Xc,'Yc:', Yc,'r:',r)

#Dibujo la geometría generada
cilindro = plt.Circle((Xc,Yc),r,fill = False)

#Cargo los parámetros del cilindro
cilindro1 = SCATTERER_parameters()
cilindro1.epsr = 2.1 #permitividad relativa. Entre [10.0, 80.0]
cilindro1.sigma = 0.0
cilindro1.f = TRANSMISOR_parameters.f #frecuencia 1.0 GHz (por defecto).
cilindro1.radio = r
cilindro1.xc = Xc
cilindro1.yc = Yc

print('Permitividad medio:',ACOPLANTE_parameters.epsr)
print('Conductividad medio:',ACOPLANTE_parameters.sigma)

print('Permitividad del cilindro:',cilindro1.epsr)
print('Conductividad del cilindro:',cilindro1.sigma)

figure, axes = plt.subplots()
plt.xlim(-0.3/2, 0.3/2)
plt.ylim(-0.3/2 , 0.3/2)
axes.set_aspect(1)
axes.add_artist(cilindro)
axes.plot(xantenas,yantenas,'ok')
#plt.show()


resolucion = 5
n = resolucion*sx/a
tx = 0
Ezfdtd,eps_data = RunMeep(cilindro1,ACOPLANTE_parameters,TRANSMISOR_parameters, tx, box,RES = 5,calibration = False)
Ezfdtdinc,eps_data_no = RunMeep(cilindro1,ACOPLANTE_parameters,TRANSMISOR_parameters, tx, box,RES = 5,calibration = True)

EzTr = np.zeros(nant)
EzTrinc = np.zeros(nant)
print(len(xantenas))

plt.figure()
extent2=[-0.3/2,0.3/2,-0.3/2,0.3/2]
plt.imshow(abs(Ezfdtd).transpose())#.transpose())extent = extent2)#cmap = 'binary')
#plt.plot(xantenas,yantenas,'ow')
plt.colorbar()

for tr in range(nant):
    print('Coordenadas de antenas:',tr+1,':(',xantenas[tr],-yantenas[tr],')')
    xRint = int(resolucion*(xantenas[tr])/a)+int(n/2) #Coordenada x antena emisora
    yRint = int(resolucion*(-yantenas[tr])/a)+int(n/2)
    EzTr[tr] = abs(Ezfdtd)[xRint,yRint]
    EzTrinc[tr] = abs(Ezfdtdinc)[xRint,yRint]
    print('Einc :',EzTrinc[tr],' tr:',tr)
    plt.plot(xRint,yRint,'ow')
    plt.text(xRint+2,yRint+2, str(tr), fontsize=12)


#plt.show()


fig10 = plt.figure(10)
ax1 = fig10.add_subplot(121)
ax1.plot(Sinc_FINAL[0,:],'ob-',label='Incidente, Tx: 1, f: '+str(f_medida[nfrec]/1e6)+' MHz')
ax1.plot(Sinc_FINAL[1,:],'xr-',label='Incidente, Tx: 2')
#ax1.plot(Sinc_FINAL[2,:],'sg-',label='Incidente, Tx: 3')
ax1.plot(Sinc_FINAL[3,:],'vy-',label='Incidente, Tx: 4')
#ax1.plot(Sinc_FINAL[6,:],'^m-',label='Incidente, Tx: 7')
ax1.plot(S_FINAL[0,:],'ob--',label='Teflon, Tx: 1, f: '+str(f_medida[nfrec]/1e6)+' MHz')
ax1.plot(S_FINAL[1,:],'xr--',label='Teflon, Tx: 2')#, f: '+str(f_medida[nfrec]/1e6)+' MHz')
#ax1.plot(S_FINAL[2,:],'sg--',label='Teflon, Tx: 3')#, f: +str(f_medida[nfrec]/1e6)+' MHz')
ax1.plot(S_FINAL[3,:],'vy--',label='Teflon, Tx: 4')#, f: '+str(f_medida[nfrec]/1e6)+' MHz')
#ax1.plot(S_FINAL[6,:],'^m--',label='Teflon, Tx: 7')#, f: '+str(f_medida[nfrec]/1e6)+' MHz')
ax1.set_ylim([-65, -25])
ax1.legend(loc="upper right")
ax1.set_xticks([0, 1, 2, 3, 4, 5, 6])
ax1.set_xticklabels(['12', '13', '14', '15', '16', '17', '18']) 

ax2 = fig10.add_subplot(122)

ax2.plot(10.0*np.log10(EzTrinc[1:]/7.5e4),'o-',label='Incidente, f: '+str(f_medida[nfrec]/1e6)+' MHz')
ax2.plot(10.0*np.log10(EzTr[1:]/7.5e4),'x--',label='Teflon')#, f: '+str(f_medida[nfrec]/1e6)+' MHz')
#ax2.set_ylim([-65, -35])
ax2.legend(loc="upper right")
ax2.set_xticks([0, 1, 2, 3, 4, 5, 6])
ax2.set_xticklabels(['12', '13', '14', '15', '16', '17', '18']) 
ax2.set_ylim([-65, -25])
plt.show()




## Ahora calculamos los coeficientes de calibración $c_{F_{ls}}^{inc}$ y $c_{F_{ls}}^{sct}$ con los valores medidos en el setup experimental (archivo: _Med_pos_1_pos_3_prueba1.txt_, son las antenas opuestas).

## In[6]:


#canonico = 'MED_NaCL_8pos_teflon_Npos_4_04-Apr-2023_15h_2m_53s_pos_completo_col.txt'#Cilindro de teflon
#datos = np.loadtxt(canonico,skiprows=1)
#f_medida = datos[:,0]
#S_medida = datos[:,1]


#fig2 = plt.figure(5)
#ax1 = fig2.add_subplot(121)
#ax1.semilogx(f_medida,S_medida,'-k')
##ax1.set_xlim([0.6e9, 1.6e9])
##ax1.set_ylabel('Parte real permitividad')
##ax1.set_xlabel('Frecuencia (Hz)')
##ax2 = fig2.add_subplot(122)
##ax2.semilogx(frec,-epsc.imag,'-b')
##ax2.set_xlim([0.6e9, 1.6e9])
##ax2.set_ylabel('Parte imaginaria permitividad')
##ax2.set_xlabel('Frecuencia (Hz)')

#plt.show()

##csct_F = abs(EzTr)/abs(10**(S_medida/20))

##print('c_F :', csct_F)


##print('start time: ', start_time)
##print('end time:   ', tm.strftime('%H:%M:%S'))


### Ahora simulamos el cilindro de Nylon descentrado con lo cual tendremos un valor teórico y, luego, vemos cuál es el valor medido y calibrado con el coeficiente $c_{F_{ls}}^{sct}$.

### In[7]:


##r = 35.5e-3/2.0
##Xc = -0.025
##Yc = -0.025

##print('Xc:', Xc,'Yc:', Yc,'r:',r)

###Dibujo la geometría generada
##cilindro = plt.Circle((Xc,Yc),r,fill = False)

###Cargo los parámetros del cilindro
##cilindro1 = SCATTERER_parameters()
##cilindro1.epsr = 4.1 #permitividad relativa. Entre [10.0, 80.0]
##cilindro1.sigma = 0.0
##cilindro1.f = TRANSMISOR_parameters.f #frecuencia 1.0 GHz (por defecto).
##cilindro1.radio = r
##cilindro1.xc = Xc
##cilindro1.yc = Yc

##print('Permitividad medio:',ACOPLANTE_parameters.epsr)
##print('Conductividad medio:',ACOPLANTE_parameters.sigma)

##print('Permitividad del cilindro:',cilindro1.epsr)
##print('Conductividad del cilindro:',cilindro1.sigma)

##figure, axes = plt.subplots()
##plt.xlim(-0.25/2, 0.25/2)
##plt.ylim(-0.25/2 , 0.25/2)
##axes.set_aspect(1)
##axes.add_artist(cilindro)
##axes.plot(xantenas,yantenas,'ok')
##plt.show()


### In[8]:


##Ezfdtd_sct,eps_data = RunMeep(cilindro1,ACOPLANTE_parameters,TRANSMISOR_parameters, tx, box,RES = 5,calibration = False)
##EzTr_sct = abs(Ezfdtd_sct)[xRint,yRint]


### Ahora comparamos con la medición:

### In[9]:


##canonico = 'Med_pos_1_pos_3_prueba2.txt'#Cilindro de teflon
##datos = np.loadtxt(canonico,skiprows=1)
##f_medida = datos[-51,0]
##S_medida_sct = datos[-51,1]
##EzTr_sct_medido = csct_F*abs(10**(S_medida_sct/20))
##print('Ez_medido: ',EzTr_sct_medido)
##print('Ez_teorico: ',EzTr_sct)


### Si calculamos el coeficiente de calibración con la simulación ($S_{FEM}^{sct}$). Estos coeficientes fueron calculados por simulación con elementos finitos.
### 
### $$c_{F_{ls}}^{FEM} = \dfrac{E_{esperado}^{sct}}{S_{FEM}^{sct}}$$
### 

### In[10]:


##S_fem = -47.4 #dB con teflón
##csct_F_FEM = abs(EzTr)/abs(10**(S_fem/20))
##S_sct_fem = -47.15
##EzTr_sct_FEM = csct_F_FEM*abs(10**(S_sct_fem/20))
##print('Ez_calibrado_fem: ',EzTr_sct_FEM)
##print('Ez_teorico: ',EzTr_sct)


### ## Conclusión
### 
### 

### In[ ]:




