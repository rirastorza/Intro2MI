#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Microwave imaging

"""

## Módulo Python: funciones_canonicos
## Author: Ramiro Irastorza 
## Email: rirastorza@iflysib.unlp.edu.ar
##

#Este módulo contiene las funciones para simulaciones de modelos canónicos

from scipy import constants as S
from scipy import special
from skimage.restoration import unwrap_phase
import numpy as N
from dolfin import *
import sys
import os
import h5py
from matplotlib import pyplot as plt

#
# - Constantes
#

pi = S.pi
eps0 = S.epsilon_0
c = S.c
mu0 = S.mu_0

#
# - parámetros globales
#
    

#
# - Clase de Parámetros del dispersor
#

class SCATTERER_parameters:
    #Cilindro
    epsr = 1.2 #permitividad relativa
    sigma = 0.0 #conductividad
    mur = 1.0
    f = 1.0e9 #frecuencia 1 GHz (por defecto).
    #epsrC = epsr + sigma/(2j*pi*f*eps0);
    radio = 0.75#*c/f #radio del cilindro
    xc = 0.0 # 0.75*c/f #radio del cilindro
    yc = 0.0 # 0.75*c/f #radio del cilindro
    #k = 2*pi*f*((epsrC*mur)**0.5)/c

class ACOPLANTE_parameters:
    epsr = 1.0  #frecuencia 1 GHz (por defecto).
    sigma = 0.0 #conductividad
    mur = 1.0
    f = 1.0e9 #frecuencia 1 GHz (por defecto).
    #epsrC = epsr + sigma/(2j*pi*f*eps0);
    #k = 2*pi*f*((epsrC*mur)**0.5)/c

class TRANSMISOR_parameters:
    f = 1.0e9  #frecuencia 1 GHz (por defecto).
    rhoS = 1.5 #*c/f #radio de transmisores
    S = 8. #cantidad de transmisores (fuentes)
    amp = 1000.0 #Amplitud de la fuente
    #k = 2*pi*f/c

def cart2pol(x,y):
    rho = (x**2.0+y**2.0)**0.5
    phi = N.arctan2(y,x)
    #phi = N.arctan(y/x)
    
    return phi,rho

#
# - Definición de funciones
#


#Para unwrapping de la fase
def campoEmeep(trans, problemname = 'tobillo2d',sx = 250.0e-3,Tx = 0, resolu = 5):    
    #Campo genereado con Meep (FDTD)
    rhoS = 1.5*c/trans.f
    filename = problemname+"-EzTx"+str(Tx)+".h5"
    f = h5py.File(filename, 'r')
    a_group_key = f.keys()[0]
    ezti = N.asarray(list(f[a_group_key]))
    a_group_key = f.keys()[1]
    eztr = N.asarray(list(f[a_group_key]))
    NN = len(eztr)
    deltaX = sx/(NN)
    Ezmeep = eztr[:,:,0] +1.0j*ezti[:,:,0]
    x = N.linspace(-len(eztr[:,:,0])*deltaX/2., len(eztr[:,:,0])*deltaX/2., len(eztr[:,:,0]))
    #Fuente
    a = 0.05 #Unidad de meep
    NN = resolu*sx/a
    xSint = int(resolu*(rhoS*N.cos(Tx*2*pi/trans.S))/a)+int(NN/2) #Coordenada x antena emisora
    ySint = int(resolu*(rhoS*N.sin(Tx*2*pi/trans.S))/a)+int(NN/2)
    EzTx = abs(Ezmeep)[xSint,ySint]
    
    image_unwrapped = unwrap_phase(N.angle(Ezmeep))
    
    phaseTx = image_unwrapped[xSint,ySint]
    
    return x,abs(Ezmeep),N.angle(Ezmeep)#image_unwrapped-phaseTx

#Para unwrapping de la fase
def campoEmeepinc(trans, problemname = 'tobillo2d',sx = 250.0e-3,Tx = 0, resolu = 5):    
    #Campo genereado con Meep (FDTD)
    rhoS = 1.5*c/trans.f
    filename = problemname+"-EzincTx"+str(Tx)+".h5"
    f = h5py.File(filename, 'r')
    a_group_key = f.keys()[0]
    ezti = N.asarray(list(f[a_group_key]))
    a_group_key = f.keys()[1]
    eztr = N.asarray(list(f[a_group_key]))
    NN = len(eztr)
    deltaX = sx/(NN)
    Ezmeep = eztr[:,:,0] +1.0j*ezti[:,:,0]
    x = N.linspace(-len(eztr[:,:,0])*deltaX/2., len(eztr[:,:,0])*deltaX/2., len(eztr[:,:,0]))
    #Fuente
    a = 0.05 #Unidad de meep
    NN = resolu*sx/a
    xSint = int(resolu*((rhoS)*N.cos(Tx*2*pi/trans.S))/a)+int(NN/2) #Coordenada x antena emisora
    ySint = int(resolu*((rhoS)*N.sin(Tx*2*pi/trans.S))/a)+int(NN/2)
    EzTx = abs(Ezmeep)[xSint,ySint]
    
    image_unwrapped = unwrap_phase(N.angle(Ezmeep))
    
    phaseTx = image_unwrapped[xSint,ySint]
    
    return x,abs(Ezmeep),N.angle(Ezmeep)#image_unwrapped-phaseTx






#Calcula onda cilíndrica debido a una línea infinita de corriente
#implementado basado en (5-119) y (5-103) en [Harrington2001]
def EINC_LINESOURCE(source_location, sensor_location,acoplante):
    epsrC = acoplante.epsr + acoplante.sigma/(2j*pi*acoplante.f*eps0)
    k = 2*pi*acoplante.f*((epsrC*acoplante.mur)**0.5)/c
    factor= -k**2./(4.*2.*pi*acoplante.f*epsrC*eps0) 
    phi_s, rho_s = cart2pol(source_location[0],source_location[1])
    phi, rho = cart2pol(sensor_location[0],sensor_location[1])
    absrho = (rho**2.+rho_s**2.-2.0*rho*rho_s*N.cos(phi-phi_s))**0.5
    #Cuidado: ver expresión Ecuación siguiente a Ec. (5-102) en Harrington
    Ezinc = factor*special.hankel2(0,k*absrho)

    return Ezinc

#Solución teórica de scattering de un cilindro centrado en (0,0) en un punto.
#CUIDADO: dentro del cilindro la solución da el campo total
#pero afuer da el campo disperso.
def EZ_CILINDRO_LINESOURCE(cilindro,acoplante,source_location, sensor_location):
    frecuencia = acoplante.f
    epsrC1 = acoplante.epsr + acoplante.sigma/(2j*pi*acoplante.f*eps0)
    omega         = 2*pi*frecuencia
    phi_s, rho_s  = cart2pol(source_location[0], source_location[1])
    phi, rho      = cart2pol(sensor_location[0], sensor_location[1])
    epsrC2 = cilindro.epsr + cilindro.sigma/(2j*pi*cilindro.f*eps0)
    k_1 = 2*pi*acoplante.f*((epsrC1*acoplante.mur)**0.5)/c
    #print(acoplante.sigma,k_1,epsrC1)
    k_2 = 2*pi*cilindro.f*((epsrC2*cilindro.mur)**0.5)/c
    x_1 = k_1*cilindro.radio
    x_2 = k_2*cilindro.radio
    
    N_max = 80
    nu    = N.arange(-N_max,N_max+1)
        
    #Ez = factor*N.sum(special.hankel2(nu,x_s)*special.jn(nu, x)*N.exp(1j*nu*(phi-phi_s)))
    factor= -k_1**2./(4.*2.*pi*frecuencia*epsrC1*eps0)     

    if (rho < rho_s): # inside source
        if (rho < cilindro.radio): # inside cylinder OK!
            
            dn_num =  k_1*special.jn(nu,x_1)*special.h2vp(nu,x_1)-k_1*special.hankel2(nu,x_1)*special.jvp(nu,x_1)
            
            dn_den =  k_1*special.jn(nu,x_2)*special.h2vp(nu,x_1)-k_2*special.hankel2(nu,x_1)*special.jvp(nu,x_2)
            
            dn     =  dn_num/dn_den
            
            Ez    = factor*N.sum(dn*special.hankel2(nu,k_1*rho_s)*special.jn(nu,k_2*rho)*N.exp(1j*nu*(phi-phi_s)))
            
        else: # outside cylinder (campo disperso) OK!
        
            cn_num =  k_1*special.jn(nu,x_2)*special.jvp(nu,x_1)-k_2*special.jn(nu,x_1)*special.jvp(nu,x_2)
            
            cn_den =  -k_1*special.jn(nu,x_2)*special.h2vp(nu,x_1)+k_2*special.hankel2(nu,x_1)*special.jvp(nu,x_2)

            cn     =  cn_num/cn_den;
            
            Ez    = factor*N.sum(cn*special.hankel2(nu,k_1*rho_s)*special.hankel2(nu,k_1*rho)*N.exp(1j*nu*(phi-phi_s)));

        end
    else: # outside source radius (es igual que adentro?) (LO MODIFIQUÉ!)
        
        cn_num =  k_1*special.jn(nu,x_2)*special.jvp(nu,x_1)-k_2*special.jn(nu,x_1)*special.jvp(nu,x_2)
        
        cn_den =  -k_1*special.jn(nu,x_2)*special.h2vp(nu,x_1)+k_2*special.hankel2(nu,x_1)*special.jvp(nu,x_2)
        
        dn     =  cn_num/cn_den;
        
        Ez = factor*N.sum(dn*special.hankel2(nu,k_1*rho_s)*special.hankel2(nu,k_1*rho)*N.exp(1j*nu*(phi-phi_s)));
            
    end
    
    return Ez

#Solución teórica de scattering de un cilindro centrado en (0,0) para una matriz de tamaño de epsilon
def EZ_CILINDRO_LINESOURCE_MATRIZ(epsilon,cilindro, acoplante,trans,Tx,deltaX):
    rhoS = 1.5*c/trans.f
    x = N.linspace(-len(epsilon)*deltaX/2., len(epsilon)*deltaX/2., len(epsilon))

    xt = (rhoS)*N.cos(Tx*2*pi/trans.S) #Coordenada x antena transmisora
    yt = (rhoS)*N.sin(Tx*2*pi/trans.S) #Coordenada y antena transmisora

    postrans = [xt,yt]

    x = N.linspace(-len(epsilon)*deltaX/2.+deltaX/2, len(epsilon)*deltaX/2.-deltaX/2, len(epsilon))
    y = N.linspace(-len(epsilon)*deltaX/2.+deltaX/2, len(epsilon)*deltaX/2.-deltaX/2, len(epsilon))
    xv, yv = N.meshgrid(x, y)

    Ezinc = N.zeros_like(epsilon,dtype=complex)
    Einc = N.zeros_like(epsilon,dtype=complex)
    Ezt = N.zeros_like(epsilon,dtype=complex)
    Ezs = N.zeros_like(epsilon,dtype=complex)

    Matriz = N.zeros_like(epsilon)

    phi_s, rho_s  = cart2pol(xt, yt)

    for nx in range(len(Ezinc[:,0])):
        for ny in range(len(Ezinc[0,:])):
            rho = (x[nx]**2.+y[ny]**2.)**0.5
            if (rho < rho_s): #Dentro del radio de la fuente
                if (rho > cilindro.radio):#Fuera del cilindro
                    Ezinc[nx,ny] = EINC_LINESOURCE([xt,yt], [x[nx],y[ny]],acoplante)
                    Einc[nx,ny] = Ezinc[nx,ny]
                    Ezs[nx,ny] = EZ_CILINDRO_LINESOURCE(cilindro,acoplante,[xt,yt], [x[nx],y[ny]],acoplante.f)
                
                else:#Dentro del cilindro
                    Ezt[nx,ny] = EZ_CILINDRO_LINESOURCE(cilindro,acoplante,[xt,yt], [x[nx],y[ny]],acoplante.f)
                    Einc[nx,ny] = EINC_LINESOURCE([xt,yt], [x[nx],y[ny]],acoplante)
            else:
                Ezinc[nx,ny] = EINC_LINESOURCE([xt,yt], [x[nx],y[ny]],acoplante)
                Ezs[nx,ny] = EZ_CILINDRO_LINESOURCE(cilindro,acoplante,[xt,yt], [x[nx],y[ny]],acoplante.f)
                Einc[nx,ny] = Ezinc[nx,ny]
    
    
    Eztheory = Ezinc+Ezt+Ezs     
    
    return x,Eztheory

#Genera la geometria para una simulación con FDTD
def generaCTL(archivoout,centro, cilindro, acoplante,trans, cilindroSI, Tx,caja,tsimulacion,RES):
  #Generacion de codigo meep
  f = open (archivoout, "w")
  f.write(";Simulacion del cilindro.\n")
  f.close()

  f = open (archivoout, "a")
  
  xc = centro[0]
  yc = centro[1]

  #Parametros de simulacion
  resol = RES
  a = 0.05 #Unidad de meep
  sx = caja[0]/a #Expresada en milimetros
  sy = caja[1]/a #Expresada en milimetros
  sigmaMeep = cilindro.sigma*a/(c*cilindro.epsr*eps0) #Convierte a la conductividad a unidades meep
  rhoS = 1.5*c/trans.f

 
  f.write("(define-param sx "+str(sx)+") ; size of cell in X direction\n")
  f.write("(define-param sy "+str(sy)+") ; size of cell in Y direction\n")
  f.write("(define-param a "+str(a)+") ; unidad meep\n")
  f.write("(define-param epseff "+str(acoplante.epsr)+") ;\n")
  f.write("(define-param eps0 "+str(eps0)+") ; \n") 
  f.write("(define-param sigeff "+str(acoplante.sigma)+");\n") 
  f.write("(define-param c "+str(c)+"); velocidad de la luz en el vacio \n")
  f.write("(define-param sigmaD (/ (/ (* sigeff a ) c) (* epseff eps0)));\n")
  f.write("(define-param resol "+str(resol)+"); resolucion de la grilla\n")
  f.write("(define-param fcen (/ (* "+str(acoplante.f)+" a) c)); \n")
  
  #Defino como material por defecto a la solucion fisiologica
  f.write("(set! default-material\n")
  f.write("      (make dielectric (epsilon epseff) (D-conductivity sigmaD)))\n")

  f.write("(set! geometry-lattice (make lattice (size sx sy)))\n")
  
  f.write("(set! force-complex-fields? true)\n")
  
  f.write("(define-param no-bone? false) ; if true, have no hay cilindro\n")
  f.write("(set! geometry\n")
  f.write("      (if no-bone?\n")
  f.write("          (list (make block (center 0 0) (size sx sy infinity)\n")# infinity
  f.write("		    (material (make dielectric (epsilon epseff) (D-conductivity sigmaD)))))\n")
  f.write("          (list \n")

  #--------------------------------------------------
  #Lectura por linea de comando para generar los .ctl
  if cilindroSI == 1:
      aa='		(make cylinder (center '+str(xc/a)+' '+str(yc/a)+') (radius '+str(cilindro.radio/a)+') (height infinity) (axis 0 0 1) (material (make dielectric (epsilon '+str(cilindro.epsr)+') (D-conductivity '+str(sigmaMeep)+'))))\n'# infinity
      f.write(aa)
  else:
      Z = 0
  
  f.write(")))\n")
    
  xt = (rhoS)*N.cos(Tx*2*pi/trans.S) #Coordenada x antena transmisora
  yt = (rhoS)*N.sin(Tx*2*pi/trans.S) #Coordenada y antena transmisora

  
  f.write("(set! sources (list\n")
  f.write("               (make source\n")
  f.write("                 (src (make continuous-src (frequency fcen)))\n")
  f.write("                 (component Ez)\n")
  f.write("                 (amplitude "+str(trans.amp)+")\n")
  f.write("                 (center "+str(xt/a)+" "+str(yt/a)+")\n")
  f.write("		 	(size 0 0)))); \n")

  f.write("(set! pml-layers (list (make pml (thickness 1))))\n")

  f.write("(set! resolution "+str(resol)+")\n")

  f.write("(if no-bone? (run-until "+str(tsimulacion)+"\n")
  f.write("	(at-beginning output-epsilon)\n")
  f.write('	(to-appended "EzincTx'+str(Tx)+'" (at-end output-efield-z))\n')
  #f.write('	(at-end output-efield-z)\n')
  f.write("        ))\n")
    
  f.write("(if (not no-bone?) (run-until "+str(tsimulacion)+"\n")
  f.write("	(at-beginning output-epsilon)\n")
  f.write('	(to-appended "EzTx'+str(Tx)+'" (at-end output-efield-z))\n')
  #f.write('	(at-end output-efield-z)\n')
  f.write("        ))\n")
    
  f.close()

  return 1

#Calcula el campo total utilizando el método de los momentos (Richmond)
def numRichmond(epsilon, cilindro, acoplante,trans,Tx,deltaX):
    #Parametros de simulacion
    a = 0.05 #Unidad de meep
    frecuencia = cilindro.f
    landa = c/frecuencia
    sx = 4.0*landa 
    sy = 4.0*landa 
    
    #fuentes
    rhoS = 1.5*c/trans.f
        
    #Grafico la permititvidad
    #filename = problemname+"-eps-000000.00.h5"
    #import h5py
    #f = h5py.File(filename, 'r')
    #print("Keys: %s" % f.keys())
    #a_group_key = f.keys()[0]
    #epsilon = list(f[a_group_key])

    #print type(epsilon),epsilon.shape
    NNNN = len(epsilon)
    deltaX = sx/(NNNN)
    d = ((2.0**0.5)*rhoS)*0.5/deltaX #da mas o menos el área donde se encuentra el dispersor

    #print 'N,deltaX,d,m:',NNNN,deltaX,d,NNNN/2-int(d/2),NNNN/2+int(d/2)

    epsilon = N.asarray(epsilon)
    #Selecciono la región de interesint(d/2)
    Li = int(NNNN/2)-int(d/2)
    Ls = int(NNNN/2)+int(d/2)

    epsilonSi = epsilon[Li:Ls,Li:Ls]
                
    #print len(epsilonSi)

    xposant = N.array([])
    yposant = N.array([])
    for n in range(0,int(trans.S)):
        xrint = ((rhoS)*N.cos(n*2*pi/trans.S))+sx/(2.0)
        yrint = ((rhoS)*N.sin(n*2*pi/trans.S))+sy/(2.0)
        xposant = N.append(xposant,xrint)
        yposant = N.append(yposant,yrint)


    #xpos = N.array([])
    #ypos = N.array([])
    x = N.linspace(Li*deltaX+0.5*deltaX, Ls*deltaX+0.5*deltaX, len(epsilonSi), endpoint=True)
    y = N.linspace(Li*deltaX+0.5*deltaX, Ls*deltaX+0.5*deltaX, len(epsilonSi), endpoint=True)
    #x = N.linspace(Li*deltaX-0.5*deltaX, Ls*deltaX-0.5*deltaX, len(epsilonSi))
    #y = N.linspace(Li*deltaX-0.5*deltaX, Ls*deltaX-0.5*deltaX, len(epsilonSi))
    xv, yv = N.meshgrid(x, y)



    ##conductividad del cilindro
    sigmaSi = 0.0*N.ones_like(epsilonSi)
    ##Cilindro
    for n in range(0,len(sigmaSi)):
        for m in range(0,len(sigmaSi)): 
            if epsilonSi[n,m] > (cilindro.epsr+acoplante.epsr)/2.0:
                sigmaSi[n,m] = cilindro.sigma
            else:
                sigmaSi[n,m] = acoplante.sigma

    #fig1 = plt.figure(1)
    #f1 = fig1.add_subplot(221)
    #cax = f1.imshow(sigmaSi)
    #fig1.colorbar(cax, )#ticks=[1.0, 40.0, 80.0]

    #plt.show()
    #-----------------------------------------------
    #Solución utilizando Ritchmon 1964 "Scattering by a Dielectric Cylinder of Arbitrary Cross Section Shape", IEEE.

    NNNN = len(epsilonSi)

    #Construcción de la matriz C según Ecs. 15 y 16
    NN = len(epsilonSi)**2 #número en que se particiona el cilindro
    MM = NN #Número de lugares donde se mide el campo, puede coincidir con el número en que se particiona el cilindro
    
    am = ((deltaX)**2.0/pi)**0.5 #radio equivalente de una celda del cilindro
    w = 2.0*pi*frecuencia #frecuencia angular
    epsb = (acoplante.epsr-acoplante.sigma*1.0j/(w*eps0))*N.ones_like(epsilonSi) #permitividad relativa del background
    kb = w*((mu0*epsb*eps0)**0.5) #numero de onda

    epsilonSiR = epsilonSi-sigmaSi*1.0j/(w*eps0)

    #print 'Número de onda:',kb[0]
    #print 'Lambda:',2*pi/(w*((54.3-1.1j/(w*epsilon_0))*epsilon_0*mu0)**0.5).real
    #print 'Grilla Richmond (lambda/10):',(2*pi/(w*((54.3-1.1j/(w*epsilon_0))*epsilon_0*mu0)**0.5).real)/10.0
    #print 'Grilla mia:',deltaX

    Cmn_pre =1.0j*pi*kb*am*0.5*(epsilonSiR-epsb)*special.jn(1,kb*am)#
    Cmn_pre2 = 1+1.0j*0.5*(epsilonSiR-epsb)*(pi*kb*am*special.hankel2(1, kb*am) -2.0j)

    Cmn_pre = N.reshape(Cmn_pre, NN)
    Cmn_pre2 = N.reshape(Cmn_pre2, NN)

    xv2 = N.reshape(xv, NN)
    yv2 = N.reshape(yv, NN)
    kb2 = N.reshape(kb, NN)
    epsb2 = N.reshape(epsb, NN)

    C = N.empty([NN, NN])+ 0j

    #Construcción de la matriz
    for n in range(0,NN):
        for m in range(0,MM):
            if m == n:
                C[n,m] = Cmn_pre2[n]
            else:
                #rho = ((xv2[n]-xv2[m])**2.0+(yv2[n]-yv2[m])**2.0)**0.5
                phi_s, rho_s = cart2pol(xv2[n],yv2[n])
                phi, rho = cart2pol(xv2[m],yv2[m])
                absrho = (rho**2.+rho_s**2.-2.0*rho*rho_s*N.cos(phi-phi_s))**0.5
                C[n,m] = Cmn_pre[n]*special.hankel2(0, kb2[n]*absrho) 

    #Construcción del término independiente
    Ei = N.empty(NN)+ 0j
    
    I = 1.0
    epsrC1 = acoplante.epsr + acoplante.sigma/(2j*pi*acoplante.f*eps0)
    k_1 = 2*pi*acoplante.f*((epsrC1*acoplante.mur)**0.5)/c
    for n in range(0,NN):
        factor= -I*k_1**2./(4.*2.*pi*acoplante.f*epsrC1*eps0) 
        phi_s, rho_s = cart2pol(xposant[Tx],yposant[Tx])
        phi, rho = cart2pol(xv2[n],yv2[n])
        absrho = (rho**2.+rho_s**2.-2.0*rho*rho_s*N.cos(phi-phi_s))**0.5
        #Cuidado: ver expresión Ecuación siguiente a Ec. (5-102) en Harrington
        Ei[n] = factor*special.hankel2(0,k_1*absrho)

    #Solución de la ecuación lineal
    x = N.linalg.solve(C, Ei)
    print N.allclose(N.dot(C, x), Ei)
    
    #Enf = N.reshape(x,(NNNN,NNNN)).T #ojo me queda transpuesta correcta
    
    En = N.reshape(x,(NNNN,NNNN))
    
    print x.shape,En.shape,type(En)
    
    #Según la ecuación 17 del trabajo
    #se puede calcular el campo disperso en cualquier punto
    Etotal = N.zeros_like(epsilon)+0.0j

    #Cilindro
    sigma = 0.0*N.ones_like(epsilon)
    
    epsbb = (acoplante.epsr-acoplante.sigma*1.0j/(w*eps0))*N.ones_like(epsilon) #permitividad relativa del background
    kbb = w*(mu0*epsbb*eps0)**0.5 #numero de onda

    #epsilonSiR = epsilonSi-sigmaSi*1.0j/(w*eps0)

    Cmn_pre =-1.0j*pi*kb*am*0.5*(epsilonSiR-epsb)*special.jn(1,kb*am)*En

    x = N.linspace(0.5*deltaX, len(epsilon)*deltaX+0.5*deltaX, len(epsilon), endpoint=True)
    y = N.linspace(0.5*deltaX, len(epsilon)*deltaX+0.5*deltaX, len(epsilon), endpoint=True)
    #x = N.linspace(0.5*deltaX, len(epsilon)*deltaX+0.5*deltaX, len(epsilon))
    #y = N.linspace(0.5*deltaX, len(epsilon)*deltaX+0.5*deltaX, len(epsilon))
    xvv, yvv = N.meshgrid(x, y)

    NN = len(epsilonSi)**2
    Cmn_pre = N.reshape(Cmn_pre, NN)
    NNN = len(epsilon)**2

    xvv2 = N.reshape(xvv, NNN)
    yvv2 = N.reshape(yvv, NNN)

    kbb2 = N.reshape(kbb, NNN)
    epsbb2 = N.reshape(epsbb, NNN)
    Etotal2 = N.reshape(Etotal, NNN)

    #Construcción del campo disperso
    for m in range(0,NNN):
        #absrho = ((xv2-xvv2[m])**2.0+(yv2-yvv2[m])**2.0)**0.5 
        phi, rho = cart2pol(xv2,yv2)
        phi_s, rho_s = cart2pol(xvv2[m],yvv2[m])
        absrho2 = (rho**2.+rho_s**2.-2.0*rho*rho_s*N.cos(phi-phi_s))**0.5
        #print('sol1:',absrho,'sol2:',absrho2)
        Etotal2[m] = N.sum(Cmn_pre*special.hankel2(0, kb2*absrho2))


    Etotal = N.reshape(Etotal2, (len(epsilon),len(epsilon)))

    #Construcción del campo incidente
    Ei = N.empty(NNN)+ 0j
    
    epsrC1 = acoplante.epsr + acoplante.sigma/(2j*pi*acoplante.f*eps0)
    k_1 = 2*pi*acoplante.f*((epsrC1*acoplante.mur)**0.5)/c

    for n in range(0,NNN):
        factor= -I*k_1**2./(4.*2.*pi*acoplante.f*epsrC1*eps0) 
        phi_s, rho_s = cart2pol(xposant[Tx],yposant[Tx])
        phi, rho = cart2pol(xvv2[n],yvv2[n])
        absrho = (rho**2.+rho_s**2.-2.0*rho*rho_s*N.cos(phi-phi_s))**0.5
        #Cuidado: ver expresión Ecuación siguiente a Ec. (5-102) en Harrington
        Ei[n] = factor*special.hankel2(0,k_1*absrho)


    Ei = N.reshape(Ei, (len(epsilon),len(epsilon)))


    print 'PROBLEMAS----------------',Ei.shape,Etotal.shape

    Etotal = Ei+Etotal
    Etotal[Li:Ls,Li:Ls] = En
    Escattered = Etotal-Ei
    
    x = N.linspace((-len(epsilon)/2)*deltaX+0.5*deltaX, (len(epsilon)/2)*deltaX+0.5*deltaX, len(epsilon), endpoint=True)
    
    return x,Etotal.T





