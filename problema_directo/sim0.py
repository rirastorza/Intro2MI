#!/usr/bin/env python
# coding: utf-8

# # Uso del módulo fordward
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
# Autores: Julián Galván & Ramiro Irastorza 
# 
# Email: rirastorza@iflysib.unlp.edu.ar
# 

# ## Módulo forward **COMPLETAR
# 
# Breve descripción del módulo. Si se ejecuta solo con la instrucción:
# 
# > python3 forward.py
# 
# A continuación se comentan varios ejemplos de simulación.
# 
# ## Ejemplo de uso para calibración
# 
# ### Conversión de parámetros $S_{12}$ a campo eléctrico $E_{z}$ y calibración
# 
# El coeficiente de transmisión entre antena transmisora (tx) y receptora (tr) se denomina parámetro $S_{12}$. Como los algoritmos de reconstrucción de imágenes siempre utilizan el campo eléctrico entonces es necesario traducir la información medida ($S_{12}$) a la teórica ($E_{z}$, en nuestro caso porque utilizamos algorimos 2D en configuración TM). Según Ostadrahimi M. y otros (2011) [enlace](https://www.researchgate.net/profile/Amer_Zakaria/publication/224257030_Analysis_of_Incident_Field_Modeling_and_IncidentScattered_Field_Calibration_Techniques_in_Microwave_Tomography/links/0912f50a51f7a51be0000000.pdf), esto se puede hacer calculando los coeficientes de calibración ($c_{F}$) en cada uno de los **receptores** para cada uno de los **transmisores**. Con el cálculo de estos coeficientes también se pueden compesar algunos otros errores debido a diferentes longitudes de camino. Se puede hacer de dos maneras:
# 
# 1. Con el campo incidente. La expresión en este caso es la siguiente:
# 
# $$c_{F_{ls}} = \dfrac{E_{esperado}^{inc}}{S_{medido}^{inc}}$$
# 
# donde $E_{esperado}^{inc}$ es el campo simulado o calculado analíticamene en cada receptor y $S_{medido}^{inc}$ el coeficiente de dispersión medido. El subíndice _ls_ es de line source.
# 
# 2. Con el campo disperso por un medio canónico, por ejemplo, cilindro centrado en (0,0) con dieléctrico conocido. Se reemplaza _inc_ por el _sct_, es decir, el campo eléctrico calculado y el coeficiente de dispersión con en el centro (_problema canónico_).
# 
# En este jupyter notebook vamos a calcular las dos formas. Vamos utilizar los $S_{medido}$ con nuestro setup experimental y además también vamos a simular los parámetros $S$ en 3D con FEM, luego compararemos con los valores obtenidos con simulación 2D de los campos eléctricos simulados con FDTD.
# 
# ### Simulación
# 
# En lo que sigue, simularemos tanto el campo incidente (sólo con el medio de acoplamiento) y el campo con el cilindro canónico (cilindro de teflón de diámetro 25 mm centrado en (0,0)). A continuación, mostramos cómo hacer la simulación con la librería _forward_.

# In[1]:


from forward import *
import time as tm
import os
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt


# Iniciamos el contador para estimar cuanto tarda la simulación del modelo.

# In[2]:


start_time = tm.strftime('%H:%M:%S')


# Se define la frecuencia, el tamaño de la caja de simulación 25 cm x 25 cm, donde está el arreglo de n antenas (en el ejemplo 4 antenas). Se crean los objetos _TRANSMISOR_parameters_ y _ACOPLANTE_parameters_ y se les asignan las propiedades. Estos objetos contienen información del arreglo de antenas y del fluido de acoplamiento, respectivamente. El acoplante, en este ejemplo, es agua con sal cuyo modelo dieléctrico es un Debye (una frecuencia de relajación) con conductividad iónica.

# In[3]:


f = 0.6e9
print('frecuencia de medición (GHz): ',f/1e9)
sx = 0.3
sy = 0.3
box = [sx,sy]

TRANSMISOR_parameters = TRANSMISOR_parameters()
TRANSMISOR_parameters.f = f
TRANSMISOR_parameters.amp = 7.5e4
TRANSMISOR_parameters.rhoS = 0.075
TRANSMISOR_parameters.S = 4.

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


# Luego se define el cilindro dispersor pasando los parámetros al objeto _SCATTERER_parameters_. El cilindro está centrado en (0,0) y es de teflon (permitividad relativa 2.1 y conductividad nula). Aquí también lo dibujamos, junto con el arreglo de antenas.

# In[4]:


#Coordenadas antenas
angulo = N.linspace(0.0, 2.0*pi, 5)
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


# Ahora simulamos, utilizando el transmisor número 2 (_tx = 2_) luego seleccionamos el campo en el receptor 0 (_t_r=0_). De esta manera obtenemos $E_{esperado}^{inc}$ y $E_{esperado}^{sct}$.

# In[5]:


resolucion = 5
n = resolucion*sx/a
tx = 2
Ezfdtd,eps_data = RunMeep(cilindro1,ACOPLANTE_parameters,TRANSMISOR_parameters, tx, box,RES = 5,calibration = False)
Ezfdtdinc,eps_data_no = RunMeep(cilindro1,ACOPLANTE_parameters,TRANSMISOR_parameters, tx, box,RES = 5,calibration = True)

tr = 0
xRint = int(resolucion*((TRANSMISOR_parameters.rhoS)*N.cos(tr*2*pi/4.))/a)+int(n/2) #Coordenada x antena emisora
yRint = int(resolucion*((TRANSMISOR_parameters.rhoS)*N.sin(tr*2*pi/4.))/a)+int(n/2)

EzTr = abs(Ezfdtd)[xRint,yRint]
EzTrinc = abs(Ezfdtdinc)[xRint,yRint]

plt.figure()
extent2=[-0.3/2,0.3/2,-0.3/2,0.3/2]
plt.imshow(abs(Ezfdtd).transpose(),extent = extent2)#cmap = 'binary')
plt.plot(xantenas,yantenas,'ow')
plt.colorbar()

#plt.show()


# Ahora calculamos los coeficientes de calibración $c_{F_{ls}}^{inc}$ y $c_{F_{ls}}^{sct}$ con los valores medidos en el setup experimental (archivo: _Med_pos_1_pos_3_prueba1.txt_, son las antenas opuestas).

# In[6]:


canonico = 'MED_NaCL_8pos_teflon_Npos_4_04-Apr-2023_15h_2m_53s_pos_completo_col.txt'#Cilindro de teflon
datos = np.loadtxt(canonico,skiprows=1)
f_medida = datos[:,0]
S_medida = datos[:,1]


fig2 = plt.figure(5)
ax1 = fig2.add_subplot(121)
ax1.semilogx(f_medida,S_medida,'-k')
#ax1.set_xlim([0.6e9, 1.6e9])
#ax1.set_ylabel('Parte real permitividad')
#ax1.set_xlabel('Frecuencia (Hz)')
#ax2 = fig2.add_subplot(122)
#ax2.semilogx(frec,-epsc.imag,'-b')
#ax2.set_xlim([0.6e9, 1.6e9])
#ax2.set_ylabel('Parte imaginaria permitividad')
#ax2.set_xlabel('Frecuencia (Hz)')

plt.show()

#csct_F = abs(EzTr)/abs(10**(S_medida/20))

#print('c_F :', csct_F)


#print('start time: ', start_time)
#print('end time:   ', tm.strftime('%H:%M:%S'))


## Ahora simulamos el cilindro de Nylon descentrado con lo cual tendremos un valor teórico y, luego, vemos cuál es el valor medido y calibrado con el coeficiente $c_{F_{ls}}^{sct}$.

## In[7]:


#r = 35.5e-3/2.0
#Xc = -0.025
#Yc = -0.025

#print('Xc:', Xc,'Yc:', Yc,'r:',r)

##Dibujo la geometría generada
#cilindro = plt.Circle((Xc,Yc),r,fill = False)

##Cargo los parámetros del cilindro
#cilindro1 = SCATTERER_parameters()
#cilindro1.epsr = 4.1 #permitividad relativa. Entre [10.0, 80.0]
#cilindro1.sigma = 0.0
#cilindro1.f = TRANSMISOR_parameters.f #frecuencia 1.0 GHz (por defecto).
#cilindro1.radio = r
#cilindro1.xc = Xc
#cilindro1.yc = Yc

#print('Permitividad medio:',ACOPLANTE_parameters.epsr)
#print('Conductividad medio:',ACOPLANTE_parameters.sigma)

#print('Permitividad del cilindro:',cilindro1.epsr)
#print('Conductividad del cilindro:',cilindro1.sigma)

#figure, axes = plt.subplots()
#plt.xlim(-0.25/2, 0.25/2)
#plt.ylim(-0.25/2 , 0.25/2)
#axes.set_aspect(1)
#axes.add_artist(cilindro)
#axes.plot(xantenas,yantenas,'ok')
#plt.show()


## In[8]:


#Ezfdtd_sct,eps_data = RunMeep(cilindro1,ACOPLANTE_parameters,TRANSMISOR_parameters, tx, box,RES = 5,calibration = False)
#EzTr_sct = abs(Ezfdtd_sct)[xRint,yRint]


## Ahora comparamos con la medición:

## In[9]:


#canonico = 'Med_pos_1_pos_3_prueba2.txt'#Cilindro de teflon
#datos = np.loadtxt(canonico,skiprows=1)
#f_medida = datos[-51,0]
#S_medida_sct = datos[-51,1]
#EzTr_sct_medido = csct_F*abs(10**(S_medida_sct/20))
#print('Ez_medido: ',EzTr_sct_medido)
#print('Ez_teorico: ',EzTr_sct)


## Si calculamos el coeficiente de calibración con la simulación ($S_{FEM}^{sct}$). Estos coeficientes fueron calculados por simulación con elementos finitos.
## 
## $$c_{F_{ls}}^{FEM} = \dfrac{E_{esperado}^{sct}}{S_{FEM}^{sct}}$$
## 

## In[10]:


#S_fem = -47.4 #dB con teflón
#csct_F_FEM = abs(EzTr)/abs(10**(S_fem/20))
#S_sct_fem = -47.15
#EzTr_sct_FEM = csct_F_FEM*abs(10**(S_sct_fem/20))
#print('Ez_calibrado_fem: ',EzTr_sct_FEM)
#print('Ez_teorico: ',EzTr_sct)


## ## Conclusión
## 
## 

## In[ ]:




