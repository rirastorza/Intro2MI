#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Microwave imaging

"""

## Módulo Python: inverseborn
## Author: Ramiro Irastorza 
## Email: rirastorza@iflysib.unlp.edu.ar
##

#Este módulo contiene las funciones para reconstrucción mediante 
#imágenes por microondas utilizando métodos Born-like

from scipy import constants as S
from scipy import special
from skimage.restoration import unwrap_phase
import numpy as N
#from dolfin import *
import sys
import os
import h5py
import xml.etree.ElementTree as ET

#
# - Constantes
#

pi = S.pi
eps0 = S.epsilon_0
c = S.c

#
# - parámetros globales
#
    
resol = 5
a = 0.05 #Unidad de meep

#
# - Clase de Parámetros del dispersor
#

class SCATTERER_parameters:
    epsr = 1.2 #permitividad relativa
    sigma = 0.0 #conductividad

class TRANSMISOR_parameters:
    f = 1.0e9  #frecuencia 1 GHz (por defecto).
    rhoS = 1.5*c/f #radio de transmisores
    S = 8. #cantidad de transmisores (fuentes)
    k = 2*pi*f/c
    

class ACOPLANTE_parameters:
    epsr = 1.0  #frecuencia 1 GHz (por defecto).
    sigma = 0.0 #conductividad
    mur = 1.0
    f = 1.0e9 #frecuencia 1 GHz (por defecto).
    epsrC = epsr + sigma/(2j*pi*f*eps0);
    k = 2*pi*f*((epsrC*mur)**0.5)/c

#class RECEPTOR_parameters:
   
    



#Para unwrapping de la fase
def campoEmeep(trans, problemname = 'tobillo2d',sx = 250.0e-3,Tx = 0, resolu = 5):    
    #Campo genereado con Meep (FDTD)
    rhoS = 1.5*c/trans.f
    filename = problemname+"-EzTx"+str(Tx)+".h5"
    f = h5py.File(filename, 'r')
    ezti = N.asarray(f[list(f.keys())[0]][:])
    eztr = N.asarray(f[list(f.keys())[1]][:])
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
    ezti = N.asarray(f[list(f.keys())[0]][:])
    eztr = N.asarray(f[list(f.keys())[1]][:])
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










# A partir de los archivos xml (generados a partir de gmsh)
#
# genera un archivo .ctl necesario para la simulación en Meep, diferencias finitas.
#
# args: problemname,archivosalida, resolucion meep, hueso, transmisor, 
# parametros dielectricos, parametros del transmisor
#    

def generaCTLdeXML(archivoin,archivoout,resol,hueso, Tx, emp,trans,tsimulacion):

    landa = c/(trans.f)
    #Archivos .xml
    regiones = archivoin+'_physical_region'
    bordes = archivoin+'_facet_region'
    tree = ET.parse(archivoin+'.xml')
    root = tree.getroot()

    indicepts = []
    xpospts = []
    ypospts = []

    for verts in root.findall("./mesh/vertices/vertex"):
        indicepts.append(verts.get('index'))
        xpospts.append(verts.get('x'))
        ypospts.append(verts.get('y'))
        
    indpts = (N.asarray(indicepts)).astype(N.int)    
    x = (N.asarray(xpospts)).astype(N.float)
    y = (N.asarray(ypospts)).astype(N.float)

    indicetrs = []
    vpos0 = []
    vpos1 = []
    vpos2 = []

    for verts in root.findall("./mesh/cells/triangle"):
        indicetrs.append(verts.get('index'))
        vpos0.append(verts.get('v0'))
        vpos1.append(verts.get('v1'))
        vpos2.append(verts.get('v2'))

    indtr = (N.asarray(indicetrs)).astype(N.int)    
    v0 = (N.asarray(vpos0)).astype(N.int)
    v1 = (N.asarray(vpos1)).astype(N.int)
    v2 = (N.asarray(vpos2)).astype(N.int)

    tree = ET.parse(regiones+'.xml')
    root = tree.getroot()

    propiedad = []

    for prop in root.findall("./mesh_function/entity"):
        propiedad.append(prop.get('value'))

    perm = (N.asarray(propiedad)).astype(N.int)#Este es dato a usar
    #--------------------------------------------
    #Traducción a .ctl de meep
    #..............................................
    f = open (archivoout, "w")
    f.write(";Simulacion del calcaneo con triangulos.\n")
    f.close()
    f = open (archivoout, "a")
    #Parametros de simulacion
    
    sx = 4.0*landa/a 
    sy = 4.0*landa/a 
    amp = 1000.0 #Amplitud de la fuente
    
    sigmaMeepH = emp.sigma*a/(c*emp.epsr*eps0) #conductividad del dispersor al estilo Meep

    f.write("(define-param sx "+str(sx)+") ; size of cell in X direction\n")
    f.write("(define-param sy "+str(sy)+") ; size of cell in Y direction\n")
    f.write("(define-param a "+str(a)+") ; unidad meep\n")
    f.write("(define-param epseff 1.0) ; permitividad de medio acoplante: vacio\n")
    f.write("(define-param eps0 "+str(eps0)+") ; \n") #permitividad del vacio
    f.write("(define-param sigeff 0.0) ;conductividad del medio acoplante: vacio\n") 
    f.write("(define-param c "+str(c)+"); velocidad de la luz en el vacio \n")
    f.write("(define-param sigmaD (/ (/ (* sigeff a ) c) (* epseff eps0))); D-conductividad a la manera meep\n")
    f.write("(define-param resol "+str(resol)+"); resolucion de la grilla\n")
    f.write("(define-param fcen (/ (* "+str(trans.f)+" a) c)); frecuencia central del pulso (en unidades meep)\n")

    #Defino como material por defecto el vacio
    f.write("(set! default-material\n")
    f.write("      (make dielectric (epsilon epseff) (D-conductivity sigmaD)))\n")

    f.write("(set! geometry-lattice (make lattice (size sx sy)))\n")

    #Nuevo!
    f.write("(set! force-complex-fields? true)\n") #prueba....

    f.write("(define-param no-bone? false) ; if true, have no hay dispersor\n")
    f.write("(set! geometry\n")
    f.write("      (if no-bone?\n")
    f.write("          (list (make block (center 0 0) (size sx sy infinity)\n")# infinity
    f.write("		    (material (make dielectric (epsilon epseff) (D-conductivity sigmaD)))))\n")
    f.write("          (list \n")

    #-------------------------------
    #lectura de todos los bloques
    #genero bloques con centroide del triangulo
    #de cada bloque y dimensión (dimx)
    dimx = 20.0e-3 
    if hueso == 1:
        for i in range(len(indtr)):
            if perm[i] == 100: #Calcaneo
                aa='		(make block (center '+str((x[v0[i]]+x[v1[i]]+x[v2[i]])/(3.0*a))+' '+str((y[v0[i]]+y[v1[i]]+y[v2[i]])/(3.0*a))+') (size '+str(dimx/a)+' '+str(dimx/a)+' infinity) (material (make dielectric (epsilon '+str(emp.epsr)+') (D-conductivity '+str(sigmaMeepH)+'))))\n'
                f.write(aa)
            else:
                Z = 0
    else:
        Z = 0

    f.write(")))\n")

    #Agrego la antena transmisora con una fuente harmónica (línea de corriente con frecuencia fija)
    xt = (trans.rhoS)*N.cos(Tx*2*pi/trans.S) #Coordenada x antena transmisora
    yt = (trans.rhoS)*N.sin(Tx*2*pi/trans.S) #Coordenada y antena transmisora
    f.write("(set! sources (list\n")
    f.write("               (make source\n")
    f.write("                 (src (make continuous-src (frequency fcen) (width 20)))\n")
    f.write("                 (component Ez)\n")
    f.write("                 (amplitude "+str(amp)+")\n")
    f.write("                 (center "+str(xt/a)+" "+str(yt/a)+")\n")
    f.write("		 	(size 0 0)))); \n")

    f.write("(set! pml-layers (list (make pml (thickness 2))))\n")

    f.write("(set! resolution "+str(resol)+")\n")

    f.write("(if no-bone? (run-until "+str(tsimulacion)+"\n")
    f.write("	(at-beginning output-epsilon)\n")
    f.write('	(to-appended "EzincTx'+str(Tx)+'" (at-end output-efield-z))\n')
    #f.write('	(at-end output-efield-z)\n')
    f.write("        ))\n")

    f.write("(if (not no-bone?) (run-until "+str(tsimulacion)+"\n")
    f.write("	(at-beginning output-epsilon)\n")
    f.write('	(to-appended "EzTx'+str(Tx)+'" (at-end output-efield-z))\n')
    f.write("        ))\n")

    f.close()
    
    return 1



# Campos eléctricos FDTD
#
# computa los campos utilizando FDTD, problema directo
#
# args:
#    

def DIRECTO_FDTD(archivoin,archivoout,resol,hueso, emp,trans,tsimulacion):
    
    print("--+--+-- compute DIRECT PROBLEM INCIDENT FIELD FDTD --+--+--")
    #Corridas de normalización Ezinc (campo incidendte)
    for m in range(int(trans.S)):
        print("Corrida sin hueso, Tx:",m)
        flagn = generaCTLdeXML(archivoin,archivoin+'.ctl',resol,0, m,emp,trans,tsimulacion)
        string = "meep no-bone?=true "+archivoin+'.ctl'
        os.system(string)
    
    #print "Corrida sin hueso, Tx:",0
    #flagn = generaCTLdeXML(archivoin,archivoin+'.ctl',resol,0, 0,emp,trans,tsimulacion)
    #string = "meep no-bone?=true "+archivoin+'.ctl'
    #os.system(string)

    
    print("--+--+-- compute DIRECT PROBLEM TOTAL FIELD FDTD --+--+--")
    #Corridas con dispersor Ez (campo total)
    for n in range(int(trans.S)):
        print("Corrida con hueso, Tx:",n)
        flagn = generaCTLdeXML(archivoin,archivoin+'.ctl',resol,hueso, n,emp,trans,tsimulacion)
        string = "meep no-bone?=false "+archivoin+'.ctl'
        os.system(string)

    #print "Corrida con hueso, Tx:",0
    #flagn = generaCTLdeXML(archivoin,archivoin+'.ctl',resol,hueso, 0,emp,trans)
    #string = "meep no-bone?=false "+archivoin+'.ctl'
    #os.system(string)
    

    #print("Graficando el mapa de permitividades...")
    #string = "h5topng -S3 "+archivoin+"-eps-000000.00.h5"
    #os.system(string)


    
    return 1

#   def EINC_TEORICO(archivoin,archivoout,resol,hueso, emp,trans):

#% Calculate the cylindrical wave due to an infinitely-long current
#% source with a unity current of 1A. This is the numerical
#% implementation of (5-119) based on (5-103) in [Harrington2001]


def EINC_TEORICO(source_location, sensor_location,acoplante):

    phi_s, rho_s = cart2pol(source_location[0], source_location[1])
    phi, rho   = cart2pol(sensor_location[0], sensor_location[1])
    x = acoplante.k*rho
    x_s = acoplante.k*rho_s
    factor= -acoplante.k**2./(4.*2.*pi*acoplante.f*acoplante.epsrC*eps0) 
    N_max = 20
    nu    = N.arange(-N_max,N_max+1)
    Ez = factor*N.sum(special.hankel2(nu,x_s)*special.jn(nu, x)*N.exp(1j*nu*(phi-phi_s)))
    
    return Ez

def cart2pol(x,y):
    rho = (x**2.0+y**2.0)**0.5
    phi = N.arctan2(y,x)
    #phi = N.arctan(y/x)
    
    return phi,rho
