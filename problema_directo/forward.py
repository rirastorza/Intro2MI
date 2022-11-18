#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Microwave imaging: solution for many forward problems

Este módulo contiene las funciones para simulaciones del problema directo
de modelos canónicos y geometrías provistas por gmsh.

Está basado en los siguientes libros y publicaciones:

-Xudong Chen, Computational Methods for Electromagnetic Inverse Scattering
-Matteo Pastorino, Microwave Imaging


Módulo Python: forward_problem
Author: Ramiro Irastorza 
Email: rirastorza@iflysib.unlp.edu.ar

"""

from scipy import constants as S
from scipy import special
#from skimage.restoration import unwrap_phase
import numpy as N
#from dolfin import *
import sys
import os
#import h5py
from matplotlib import pyplot as plt
import meep as mp


#
# - Constantes
#

pi = S.pi
eps0 = S.epsilon_0
c = S.c
mu0 = S.mu_0

a = 0.005 #Meep unit

#
# - Clases de parámetros del dispersor, acoplante, y transmisor. 
#
class SCATTERER_parameters:
    #Cilindro
    epsr = 1.2 #permitividad relativa
    sigma = 0.0 #conductividad
    mur = 1.0
    f = 1.0e9 #frecuencia 1 GHz (por defecto).
    #epsrC = epsr + sigma/(2j*pi*f*eps0);
    radio = 0.25#*c/f #radio del cilindro
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
    rhoS = 0.075 #*c/f #radio de transmisores
    S = 16. #cantidad de transmisores (fuentes)
    amp = 1000.0 #Amplitud de la fuente
    #k = 2*pi*f/c

#
# - Función para cambio de coordenadas cartesianas a polares
#
def cart2pol(x,y):
    rho = (x**2.0+y**2.0)**0.5
    phi = N.arctan2(y,x)
    #phi = N.arctan(y/x)
    return phi,rho


#
# - Definición de funciones 
#

#
# - Definición de funciones canónicas
#

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

        #end
    else: # outside source radius (es igual que adentro?) (LO MODIFIQUÉ!)

        cn_num =  k_1*special.jn(nu,x_2)*special.jvp(nu,x_1)-k_2*special.jn(nu,x_1)*special.jvp(nu,x_2)

        cn_den =  -k_1*special.jn(nu,x_2)*special.h2vp(nu,x_1)+k_2*special.hankel2(nu,x_1)*special.jvp(nu,x_2)

        dn     =  cn_num/cn_den;

        Ez = factor*N.sum(dn*special.hankel2(nu,k_1*rho_s)*special.hankel2(nu,k_1*rho)*N.exp(1j*nu*(phi-phi_s)));

    #end

    return Ez

#Solución teórica de scattering de un cilindro centrado en (0,0) para una matriz de tamaño de epsilon
def EZ_CILINDER_LINESOURCE_MATRIZ(epsilon,cilindro, acoplante,trans,Tx,deltaX):
    rhoS = trans.rhoS
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
                    #Einc[nx,ny] = Ezinc[nx,ny]
                    Ezs[nx,ny] = EZ_CILINDRO_LINESOURCE(cilindro,acoplante,[xt,yt], [x[nx],y[ny]])

                else:#Dentro del cilindro
                    Ezt[nx,ny] = EZ_CILINDRO_LINESOURCE(cilindro,acoplante,[xt,yt], [x[nx],y[ny]])
                    #Einc[nx,ny] = EINC_LINESOURCE([xt,yt], [x[nx],y[ny]],acoplante)
            else:
                Ezinc[nx,ny] = EINC_LINESOURCE([xt,yt], [x[nx],y[ny]],acoplante)
                Ezs[nx,ny] = EZ_CILINDRO_LINESOURCE(cilindro,acoplante,[xt,yt], [x[nx],y[ny]])
                #Einc[nx,ny] = Ezinc[nx,ny]


    Eztheory = Ezinc+Ezt+Ezs

    return x,Eztheory



#
# - Funciones numéricas (FDTD)
#

#
# - Función numérica con FDTD utilizando software meep
#

def RunMeep(cilindro, acoplante,trans, Tx,caja,RES = 5,calibration = False):
    
    a = 0.005 #Meep unit
    res = RES # pixels/a
    dpml = 1 
    
    sx = caja[0]/a 
    sy = caja[1]/a 
    
    print('sxa: ',sx,'sxa: ',sy)
    
    #rhoS = 1.5*c/trans.f
    
    fcen = trans.f*(a/c)  # pulse center frequency
    sigmaBackgroundMeep = acoplante.sigma*a/(c*acoplante.epsr*eps0)    
    sigmaCylinderMeep = cilindro.sigma*a/(c*cilindro.epsr*eps0)
    
    materialBackground = mp.Medium(epsilon=acoplante.epsr, D_conductivity= sigmaBackgroundMeep) # Background dielectric properties at operation frequency
    materialCilindro = mp.Medium(epsilon= cilindro.epsr, D_conductivity= sigmaCylinderMeep) # Cylinder dielectric properties at operation frequency
   
    default_material = materialBackground

    #Simulation box and elements
    cell = mp.Vector3(sx,sy,0)
    pml_layers = [mp.PML(dpml)]
    
    if calibration:#el cilindro del centro es Background
        geometry = [mp.Cylinder(material=materialBackground, radius=cilindro.radio/a, height=mp.inf, center=mp.Vector3(cilindro.xc/a,cilindro.yc/a,0))]
    else:#el cilindro del centro es la muestra
        geometry = [mp.Cylinder(material=materialCilindro, radius=cilindro.radio/a, height=mp.inf, center=mp.Vector3(cilindro.xc/a,cilindro.yc/a,0))] 
    
    
    xt = (trans.rhoS)*N.cos(Tx*2*pi/trans.S) #Coordenada x antena transmisora
    yt = (trans.rhoS)*N.sin(Tx*2*pi/trans.S) #Coordenada y antena transmisora
    
    #amp = 1000
    sources = [mp.Source(mp.ContinuousSource(frequency=fcen),component = mp.Ez,center = mp.Vector3(xt/a,yt/a,0.0), amplitude = trans.amp,size=mp.Vector3(0.0,0.0,mp.inf))]
    
    sim = mp.Simulation(cell_size=cell, sources=sources, resolution=res, default_material=default_material, eps_averaging=False, geometry=geometry,boundary_layers=pml_layers,force_complex_fields = True)
    

    nt = 600

    sim.run(until=nt)
    
    #sim.run(until_after_sources=mp.stop_when_dft_decayed())

    eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
    ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
    
    return ez_data,eps_data


#
# - Calculado con el flujo
# Basado en: https://meep.readthedocs.io/en/latest/Python_Tutorials/Basics/#mie-scattering-of-a-lossless-dielectric-sphere
def RunMeep_flux(cilindro, acoplante,trans, Tx,Tr,caja,RES = 5,calibration = False):

    a = 0.005 #Meep unit
    res = RES # pixels/a
    dpml = 1

    sx = caja[0]/a
    sy = caja[1]/a

    #print('sxa: ',sx,'sxa: ',sy)

    fcen = trans.f*(a/c)  # pulse center frequency
    dfrq = 500e6*(a/c)
    sigmaBackgroundMeep = acoplante.sigma*a/(c*acoplante.epsr*eps0)
    sigmaCylinderMeep = cilindro.sigma*a/(c*cilindro.epsr*eps0)

    materialBackground = mp.Medium(epsilon=acoplante.epsr, D_conductivity= sigmaBackgroundMeep) # Background dielectric properties at operation frequency
    materialCilindro = mp.Medium(epsilon= cilindro.epsr, D_conductivity= sigmaCylinderMeep) # Cylinder dielectric properties at operation frequency

    default_material = materialBackground

    #Simulation box and elements
    cell = mp.Vector3(sx,sy,0)
    pml_layers = [mp.PML(dpml)]

    radio_antena = 0.5e-3
    xr = (trans.rhoS)*N.cos((Tr)*2*pi/trans.S) #Coordenada x antena transmisora
    yr = (trans.rhoS)*N.sin((Tr)*2*pi/trans.S) #Coordenada y antena transmisora

    if calibration:#el cilindro del centro es Background
        geometry = [mp.Cylinder(material=materialBackground, radius=cilindro.radio/a, height=mp.inf, center=mp.Vector3(cilindro.xc/a,cilindro.yc/a,0)),
                    mp.Cylinder(center = mp.Vector3(xr/a,yr/a,0), radius=radio_antena/a,height=mp.inf,material=mp.metal)]#receptor
    else:#el cilindro del centro es la muestra
        geometry = [mp.Cylinder(material=materialCilindro, radius=cilindro.radio/a, height=mp.inf, center=mp.Vector3(cilindro.xc/a,cilindro.yc/a,0)),
                    mp.Cylinder(center = mp.Vector3(xr/a,yr/a,0), radius=radio_antena/a,height=mp.inf,material=mp.metal)]#receptor


    xt = (trans.rhoS)*N.cos(Tx*2*pi/trans.S) #Coordenada x antena transmisora
    yt = (trans.rhoS)*N.sin(Tx*2*pi/trans.S) #Coordenada y antena transmisora

    #amp = 1000
    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=dfrq),
                                           component = mp.Ez,
                                           center = mp.Vector3(xt/a,yt/a,0.0),
                                           amplitude = trans.amp,
                                           size=mp.Vector3(0.0,0.0,mp.inf))]

    sim = mp.Simulation(cell_size=cell, sources=sources, resolution=res, default_material=default_material, eps_averaging=False, geometry=geometry,boundary_layers=pml_layers,)#force_complex_fields = True)

    nfrq = 100
    radio_receptor = 2.0e-3
    # Flujo en una caja en transmisor
    flux_box_transmisor = sim.add_flux(fcen, 0, 1,
                        mp.FluxRegion(center=mp.Vector3(xt/a+radio_receptor/a, yt/a,0), size=mp.Vector3(0.0,2*radio_receptor/a,0.0), weight=+1),
                        mp.FluxRegion(center=mp.Vector3(xt/a-radio_receptor/a, yt/a,0), size=mp.Vector3(0.0,2*radio_receptor/a,0.0), weight=-1),
                        mp.FluxRegion(center=mp.Vector3(xt/a, yt/a+radio_receptor/a,0), size=mp.Vector3(2*radio_receptor/a,0.0,0.0), weight=+1),
                        mp.FluxRegion(center=mp.Vector3(xt/a, yt/a-radio_receptor/a,0), size=mp.Vector3(2*radio_receptor/a,0.0,0.0), weight=-1))

    ## Flujo en una caja en receptor
    #flux_box_receptor = sim.add_flux(fcen,dfrq, nfrq,
                        #mp.FluxRegion(center=mp.Vector3(xr/a+radio_receptor/a, yr/a,0), size=mp.Vector3(0.0,2*radio_receptor/a,0.0), weight=-1),
                        #mp.FluxRegion(center=mp.Vector3(xr/a-radio_receptor/a, yr/a,0), size=mp.Vector3(0.0,2*radio_receptor/a,0.0), weight=+1),
                        #mp.FluxRegion(center=mp.Vector3(xr/a, yr/a+radio_receptor/a,0), size=mp.Vector3(2*radio_receptor/a,0.0,0.0), weight=-1),
                        #mp.FluxRegion(center=mp.Vector3(xr/a, yr/a-radio_receptor/a,0), size=mp.Vector3(2*radio_receptor/a,0.0,0.0), weight=+1))

# Flujo en una caja en receptor
    flux_box_receptor = sim.add_flux(fcen, 0, 1,
                        mp.FluxRegion(center=mp.Vector3(xr/a-radio_receptor/a, yr/a,0), size=mp.Vector3(0.0,2*radio_receptor/a,0.0), weight=-1),
                        mp.FluxRegion(center=mp.Vector3(xr/a+radio_receptor/a, yr/a,0), size=mp.Vector3(0.0,2*radio_receptor/a,0.0), weight=+1),
                        mp.FluxRegion(center=mp.Vector3(xr/a, yr/a+radio_receptor/a,0), size=mp.Vector3(2*radio_receptor/a,0.0,0.0), weight=+1),
                        mp.FluxRegion(center=mp.Vector3(xr/a, yr/a-radio_receptor/a,0), size=mp.Vector3(2*radio_receptor/a,0.0,0.0), weight=-1))

    print('Transmisor:')
    print('----------------------')
    print(xt/a, yt/a)
    print(xt/a+radio_receptor/a, yt/a)
    print('----------------------')

    print('Receptor:')
    print('----------------------')
    print(xr/a, yr/a)
    print(xr/a-radio_receptor/a, yr/a)
    print('----------------------')

    nt = 1500

    sim.run(until=nt)
    #sim.run(until_after_sources=mp.stop_when_dft_decayed())

    eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
    ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)

    freqs = mp.get_flux_freqs(flux_box_transmisor)
    near_flux_transmisor = mp.get_fluxes(flux_box_transmisor)
    near_flux_receptor = mp.get_fluxes(flux_box_receptor)


    return ez_data,eps_data,freqs,near_flux_transmisor,near_flux_receptor


def RunMeep2(archivoout, cilindro1,cilindro2, acoplante,trans, Tx,caja,RES = 5,calibration = False):


    res = RES # pixels/a
    dpml = 1

    sx = caja[0]/a
    sy = caja[1]/a

    print('sxa: ',sx,'sxa: ',sy)

    #rhoS = tran1.5*c/trans.f

    fcen = trans.f*(a/c)  # pulse center frequency
    sigmaBackgroundMeep = acoplante.sigma*a/(c*acoplante.epsr*eps0)
    sigmaCylinderMeep = cilindro1.sigma*a/(c*cilindro1.epsr*eps0)
    sigmaCylinderMeep2 = cilindro2.sigma*a/(c*cilindro2.epsr*eps0)

    materialBackground = mp.Medium(epsilon=acoplante.epsr, D_conductivity= sigmaBackgroundMeep) # Background dielectric properties at operation frequency
    materialCilindro = mp.Medium(epsilon= cilindro1.epsr, D_conductivity= sigmaCylinderMeep) # Cylinder dielectric properties at operation frequency
    materialCilindro2 = mp.Medium(epsilon= cilindro2.epsr, D_conductivity= sigmaCylinderMeep2) # Cylinder dielectric properties at operation frequency

    default_material = materialBackground

    #Simulation box and elements
    cell = mp.Vector3(sx,sy,0)
    pml_layers = [mp.PML(dpml)]

    if calibration:#el cilindro1 del centro es Background
        geometry = [mp.Cylinder(material=materialBackground, radius=cilindro1.radio/a, height=mp.inf, center=mp.Vector3(cilindro1.xc/a,cilindro1.yc/a,0))]
    else:#el cilindro1 del centro es la muestra
        geometry = [mp.Cylinder(material=materialCilindro, radius=cilindro1.radio/a, height=mp.inf, center=mp.Vector3(cilindro1.xc/a,cilindro1.yc/a,0)),
                    mp.Cylinder(material=materialCilindro2, radius=cilindro2.radio/a, height=mp.inf, center=mp.Vector3(cilindro2.xc/a,cilindro2.yc/a,0))]


    xt = (trans.rhoS)*N.cos(Tx*2*pi/trans.S) #Coordenada x antena transmisora
    yt = (trans.rhoS)*N.sin(Tx*2*pi/trans.S) #Coordenada y antena transmisora

    #amp = 1000
    sources = [mp.Source(mp.ContinuousSource(frequency=fcen),component = mp.Ez,center = mp.Vector3(xt/a,yt/a,0.0), amplitude = trans.amp,size=mp.Vector3(0.0,0.0,mp.inf))]

    sim = mp.Simulation(cell_size=cell, sources=sources, resolution=res, default_material=default_material, eps_averaging=False, geometry=geometry,boundary_layers=pml_layers,force_complex_fields = True)

    nt = 2000

    if calibration:#el cilindro1 del centro es agua
        #sim.run(mp.at_beginning(mp.output_epsilon),
            #mp.in_volume(mp.Volume(center=mp.Vector3(0,0,0), size=mp.Vector3(sx,sy,0)), mp.to_appended(problemname+"ez", mp.at_every(1.0, mp.output_efield_z))),
            #until=nt)

        sim.run(until=nt)
    else:
        #sim.run(mp.at_beginning(mp.output_epsilon),
            #mp.in_volume(mp.Volume(center=mp.Vector3(0,0,0), size=mp.Vector3(sx,sy,0)), mp.to_appended(problemname+"ez", mp.at_every(1.0, mp.output_efield_z))),
            #until=nt)
        sim.run(until=nt)

    eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)


    ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)

    #plt.figure()
    #plt.plot(abs(ez_data).transpose()[int(len(ez_data)/2),:],'.')# interpolation='spline36', cmap='binary')
    #plt.axis('off')
    #plt.show()


    #plt.figure()
    #plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
    #plt.imshow(abs(ez_data).transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9)
    #plt.axis('off')
    #plt.show()
    #ezt = sim.get_array(center=mp.Vector3(xct1, yct1,0), size=mp.Vector3(0,0,0), component=mp.Ez)
    #ezr = sim.get_array(center=mp.Vector3(xcr2, ycr2,0), size=mp.Vector3(0,0,0), component=mp.Ez)
    return ez_data,eps_data



if __name__ == '__main__':
    
    #Graficos
    landa = c/TRANSMISOR_parameters.f
    sx = 0.25 #4.0*landa #(aproximadamente 1.2 m)
    sy = 0.25 #4.0*landa
    Tx = 0
    box = [sx,sy]
    problemname = "Eincidente"
            
    TRANSMISOR_parameters.f = 1.1e9
    TRANSMISOR_parameters.amp = 7500.
    TRANSMISOR_parameters.rhoS = 0.075
    TRANSMISOR_parameters.S = 16.

    ACOPLANTE_parameters.f = 1.1e9
    ACOPLANTE_parameters.epsr = 28.6  #frecuencia 1 GHz (por defecto).
    ACOPLANTE_parameters.sigma = 1.264
   
    cilindro1 = SCATTERER_parameters()
    cilindro1.epsr = 10 #permitividad relativa. Entre [10.0, 80.0]
    cilindro1.sigma = 0.4 #conductividad. Entre [0.40, 1.60]
    cilindro1.f = 1.1e9 #frecuencia 1 GHz (por defecto).
    cilindro1.radio = 0.025
    cilindro1.xc = 0.04
    cilindro1.yc = 0.01
    
    cilindro2 = SCATTERER_parameters()
    cilindro2.epsr = 40 #permitividad relativa. Entre [10.0, 80.0]
    cilindro2.sigma = 0.8 #conductividad. Entre [0.40, 1.60]
    cilindro2.f = 1.1e9 #frecuencia 1 GHz (por defecto).
    cilindro2.radio = 0.02
    cilindro2.xc = 0.04
    cilindro2.yc = 0.008
    
    
    resolucion = 5
    n = resolucion*sx/a
    Tx = N.arange(16)
    Tr = N.arange(16)
    EzTr = N.zeros((16,16))
    for tx in Tx:
        Ezfdtd,eps_data = RunMeep2(problemname,cilindro1,cilindro2,ACOPLANTE_parameters,TRANSMISOR_parameters, tx, box,calibration = False)
        Ezfdtdinc,eps_data_no = RunMeep2(problemname,cilindro1,cilindro2,ACOPLANTE_parameters,TRANSMISOR_parameters, tx, box,calibration = True)
        for tr in Tr:
            xSint = int(resolucion*((0.15/2)*N.cos(tr*2*pi/16.))/a)+int(n/2) #Coordenada x antena emisora
            ySint = int(resolucion*((0.15/2)*N.sin(tr*2*pi/16.))/a)+int(n/2)
            EzTr[tx,tr] = 20.*N.log10(abs(abs(Ezfdtd)[xSint,ySint]-abs(Ezfdtdinc)[xSint,ySint])/abs(Ezfdtdinc)[xSint,ySint])

        print('Campo en emisor:',EzTr[tx,tx])
            
    plt.figure()
    plt.imshow(EzTr, cmap='binary')
    #plt.imshow(abs(ez_data).transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9)
    plt.axis('off')
    plt.show()
    
    plt.figure()
    plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='binary')
    plt.axis('off')
    plt.show()
    


