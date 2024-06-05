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
    """
    Clase que define los parámetros de un dispersor cilíndrico para simulaciones electromagnéticas.
    
    Esta clase define los parámetros y propiedades de un dispersor cilíndrico que se utilizarán
    en simulaciones electromagnéticas. Los parámetros como la permitividad relativa, conductividad,
    frecuencia, radio y coordenadas del cilindro se pueden ajustar. Además, se puede especificar
    una malla preexistente para ser utilizada en las simulaciones. La existencia de la malla se
    verifica al instanciar la clase y se almacena en la bandera booleana `thereIsMesh`.
    """
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
    
    mesh = 'valor.msh' #valor por default para la variable mesh (malla) 
    thereIsMesh = True #bandera booleana para comprobar si existe la malla

    def __init__(self, mesh='valor.msh'):
        self.mesh = mesh
        try:
            with open(mesh, "r") as archivo: #con open() compruebo si en la ruta de la malla pasada 
                contenido = archivo.read()

        except FileNotFoundError:
            SCATTERER_parameters.thereIsMesh = False



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

class RECEPTOR_parameters:
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

def RunMeep(cilindro, acoplante,trans, Tx,caja,RES = 5,calibration = False, unit=0.01):
    
    a = unit #Meep unit
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
    

    nt = 3000

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

    nt = 3000

    sim.run(until=nt)
    #sim.run(until_after_sources=mp.stop_when_dft_decayed())

    eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
    ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)

    freqs = mp.get_flux_freqs(flux_box_transmisor)
    near_flux_transmisor = mp.get_fluxes(flux_box_transmisor)
    near_flux_receptor = mp.get_fluxes(flux_box_receptor)


    return ez_data,eps_data,freqs,near_flux_transmisor,near_flux_receptor


def RunMeep2(cilindro1, cilindro2, acoplante, trans, Tx, caja, RES=5, calibration=False, unit=0.005):
    
    a = unit #Meep unit
    res = RES  # pixels/a
    dpml = 1

    sx = caja[0] / a
    sy = caja[1] / a

    print('sxa: ', sx, 'sxa: ', sy)

    # rhoS = tran1.5*c/trans.f

    fcen = trans.f * (a / c)  # pulse center frequency
    sigmaBackgroundMeep = acoplante.sigma * a / (c * acoplante.epsr * eps0)
    sigmaCylinderMeep = cilindro1.sigma * a / (c * cilindro1.epsr * eps0)
    sigmaCylinderMeep2 = cilindro2.sigma * a / (c * cilindro2.epsr * eps0)

    materialBackground = mp.Medium(epsilon=acoplante.epsr,
                                   D_conductivity=sigmaBackgroundMeep)  # Background dielectric properties at operation frequency
    materialCilindro = mp.Medium(epsilon=cilindro1.epsr,
                                 D_conductivity=sigmaCylinderMeep)  # Cylinder dielectric properties at operation frequency
    materialCilindro2 = mp.Medium(epsilon=cilindro2.epsr,
                                  D_conductivity=sigmaCylinderMeep2)  # Cylinder dielectric properties at operation frequency

    default_material = materialBackground

    # Simulation box and elements
    cell = mp.Vector3(sx, sy, 0)
    pml_layers = [mp.PML(dpml)]

    if calibration:  # el cilindro1 del centro es Background
        geometry = [mp.Cylinder(material=materialBackground, radius=cilindro1.radio / a, height=mp.inf,
                                center=mp.Vector3(cilindro1.xc / a, cilindro1.yc / a, 0))]
    else:  # el cilindro1 del centro es la muestra
        geometry = [mp.Cylinder(material=materialCilindro, radius=cilindro1.radio / a, height=mp.inf,
                                center=mp.Vector3(cilindro1.xc / a, cilindro1.yc / a, 0)),
                    mp.Cylinder(material=materialCilindro2, radius=cilindro2.radio / a, height=mp.inf,
                                center=mp.Vector3(cilindro2.xc / a, cilindro2.yc / a, 0))]

    xt = (trans.rhoS) * N.cos(Tx * 2 * pi / trans.S)  # Coordenada x antena transmisora
    yt = (trans.rhoS) * N.sin(Tx * 2 * pi / trans.S)  # Coordenada y antena transmisora

    sources = [mp.Source(mp.ContinuousSource(frequency=fcen), component=mp.Ez, center=mp.Vector3(xt / a, yt / a, 0.0),
                         amplitude=trans.amp, size=mp.Vector3(0.0, 0.0, mp.inf))]

    sim = mp.Simulation(cell_size=cell, sources=sources, resolution=res, default_material=default_material,
                        eps_averaging=False, geometry=geometry, boundary_layers=pml_layers, force_complex_fields=True)

    nt = 600

    sim.run(until=nt)

    eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
    ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)

    return ez_data, eps_data



#
# - Funciones numéricas (MoM)
#

#
# - Función numérica con MoM discretizando según Richmond
#

def RunMoM(cilindro, acoplante,trans,receptor,size_doi = 2,Tx = 1 ,RES = 40):
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
    X = R_obs*N.cos(phi) # 1 x Ns % x coordinates of receiving antennas
    Y = R_obs*N.sin(phi) # 1 x Ns % y coordinates of receiving antennas
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

    matrizCtes = N.zeros((M**2,M**2),dtype = complex)

    Xv = x.reshape((M**2,1))
    Yv = y.reshape((M**2,1))
    ji = epsono_r.reshape((M**2,1))-1
    tau = 1j*2*pi*freq*ji #object function or scattering potential
    H = N.zeros((M**2,M**2),dtype = complex)
    for mm in range(len(matrizCtes)):
        for nn in range(len(matrizCtes)):
            if mm == nn:
                matrizCtes[mm,nn] = (ji[nn])*cte
                #H[mm,nn] = cte
            else:
                R = N.sqrt((Xv[mm]-Xv[nn])**2+(Yv[mm]-Yv[nn])**2)
                matrizCtes[mm,nn] = (ji[nn])*(1j/2)*pi*k0*cellrad*special.jv(1,k0*cellrad)*special.hankel2(0,k0*R)
                #H[mm,nn] = (1j/2)*pi*k0*cellrad*special.jv(1,k0*cellrad)*special.hankel2(0,k0*R)

    N.set_printoptions(precision=3)

    D = N.eye(M**2,M**2)

    A = D+matrizCtes

    #Incident wave (linea de corriente)
    absrho = ((x.T.flatten()-XT)**2+(y.T.flatten()-YT)**2)**0.5
    E_inc = (-2*pi*freq*mu0/4*special.hankel2(0,k0*absrho)).T.reshape((M**2,1))
    #print(E_inc.shape)
    b = E_inc.reshape((M**2,1))
    #Solución de la ecuación lineal
    Et = N.linalg.solve(A, b)

    #Et2 = N.linalg.solve(N.matmul(H,N.diag(tau))+D, b)
    return Et.reshape((M,M)), epsono_r


#
# - Función numérica con MoM discretizando según Chen
#

def RunMoM_2(cilindro, acoplante,trans,caja,RES = 150):
    #Función que implementa el Método de los Momentos
    #DOCUMENTAR!!!
    
    freq = acoplante.f
    epsr_b = acoplante.epsr
    sigma_b = acoplante.sigma
    epsrCb = epsr_b + sigma_b/(2j*pi*freq*eps0)
    kb = 2*pi*freq*((epsrCb)**0.5)/c
    k0 = 2*pi*freq/c 
    landa0 = 2*pi/k0
    imp0 = 120*pi 

    size_DOI = caja[0] # size of DOI (square)
    Ni = trans.S # number of incidence
    Ns = trans.S # number of receiving antennas
    theta = N.linspace(0,2*pi-2*pi/Ni, num= int(Ni), endpoint=True) # angle of incidence
    phi = 2*pi*N.linspace(0,(Ns-1)/Ns,num= int(Ns), endpoint=True) # 1 x Ns | angle of receiving antennas
    R_obs = trans.rhoS # radius of the circle formed by receiving antennas
    X = R_obs*N.cos(phi) # 1 x Ns % x coordinates of receiving antennas
    Y = R_obs*N.sin(phi) # 1 x Ns % y coordinates of receiving antennas
    sigma_c = cilindro.sigma
    epsono_r_c = cilindro.epsr + sigma_c/(2j*pi*freq*eps0) # the constant relative permittivity of the object
    
    #Positions of the cells 
    M = RES # the square containing the object has a dimension of MxM
    d = size_DOI/M #the nearest distance of two cell centers
    
    tx = d*N.linspace(-(M-1)/2,(M-1)/2,num=M,endpoint = True) #((-(M-1)/2):1:((M-1)/2))*d # 1 x M
    ty = d*N.linspace((M-1)/2,-(M-1)/2,num=M,endpoint = True) #((-(M-1)/2):1:((M-1)/2))*d # 1 x M
    x, y = N.meshgrid(tx, ty)# M x M
    celldia = 2*N.sqrt(d**2/pi) # diameter of cells
    cellrad = celldia/2 #radius of cells

    #Relative permittivity of each cell
    r_cilinder = cilindro.radio
    epsono_r = epsrCb*N.ones((M,M),dtype = complex)
    epsono_r[(x-0.0)**2+(y-0.0)**2 <= r_cilinder**2] = epsono_r_c

    X_dif,Y_dif = N.meshgrid(d*N.linspace(1-M,M-1,num=2*M-1,endpoint = True),d*N.linspace(1-M,M-1,num=2*M-1,endpoint = True))
    R = N.sqrt(X_dif**2+Y_dif**2) # (2M-1) x (2M-1)

    ZZ = -imp0*pi*cellrad/2*special.jv(1,kb*cellrad)*special.hankel1(0,kb*R) #(2M-1) x (2M-1)
    ZZ[M-1,M-1] = -imp0*pi*cellrad/2*special.hankel1(1,kb*cellrad)-1j/(2*pi/landa0/(imp0)) # 1 x 1

    Z = N.zeros((2*M-1,2*M-1),dtype = complex)
    Z[:M,:M] = ZZ[(M-1):(2*M-1),(M-1):(2*M-1)]
    Z[M:(2*M-1),M:(2*M-1)] = ZZ[:(M-1),:(M-1)]
    Z[:M,M:(2*M-1)] = ZZ[M-1:(2*M-1),:(M-1)]
    Z[M:(2*M-1),:M] = ZZ[:(M-1),(M-1):(2*M-1)]
    
    rho_s = R_obs
    phi_s = theta.T.flatten()
    phi, rho = cart2pol(x.T.flatten(),y.T.flatten())
    absrho = N.zeros((M**2,int(Ni)),dtype = complex)
    for mm,angulo in enumerate(phi_s):
        absrho[:,mm] = (rho**2.+rho_s**2.-2.0*rho*rho_s*N.cos(phi-angulo))**0.5
        
    #Cuidado: ver expresión Ecuación siguiente a Ec. (5-102) en Harrington
    #con I = 1 (fuente de corriente) o pag 60 Pastorino
    E_inc = -2*pi*freq*mu0/4*special.hankel2(0,kb*absrho)
    
    b = (-1j*2*pi/(landa0*imp0))*N.tile((epsono_r.T.flatten()-epsrCb).reshape((M**2,1)),(1,int(Ni)))*E_inc # M^2 x Ni
    
    #Using conjugate-gradient method
    N.random.seed(0)
    Jo = N.random.randn(M**2,int(Ni))+1j*N.random.randn(M**2,int(Ni)) # M^2 x Ni
    go = AH(A(Jo,Z,M,landa0,epsono_r,epsrCb)-b,Z,M,landa0,epsono_r,epsrCb)
    po = -go

    niter = 200
    for n in range(niter):
        alphao = -N.sum(N.conj(A(po,Z,M,landa0,epsono_r,epsrCb))*(A(Jo,Z,M,landa0,epsono_r,epsrCb)-b),axis=0)/N.linalg.norm(A(po,Z,M,landa0,epsono_r,epsrCb).reshape((M**2*int(Ni),1)))**2 # 1 x Ni
        J = Jo+N.tile(alphao,(M**2,1))*po # M^2 x Ni
        g = AH(A(J,Z,M,landa0,epsono_r,epsrCb)-b,Z,M,landa0,epsono_r,epsrCb)# % M^2 x Ni
        betao = N.sum(N.conj(g)*(g-go),axis = 0)/N.sum(abs(go)**2,axis = 0)# 1 x Ni
        p = -g+N.tile(betao,(M**2,1))*po#  M^2 x N

        po = p # M^2 x Ni
        Jo = J # M^2 x Ni
        go = g # M^2 x Ni

    # Generate Scatterd E field
    # We assume that the scattered field is measured at the circle which is
    # centered at the original point with a radius equal to 3
    #
    X_obs = N.tile(X.T,(M*M,1)).T# Ns x M^2
    Y_obs = N.tile(Y.T,(M*M,1)).T# Ns x M^2

    R = N.sqrt((X_obs-N.tile(x.reshape((M*M,1),order = 'F').T,(int(Ns),1)))**2+(Y_obs-N.tile(y.reshape((M*M,1),order = 'F').T,(int(Ns),1)))**2) # Ns x M^2

    ZZ = -imp0*pi*cellrad/2*special.jv(1,kb*cellrad)*special.hankel1(0,kb*R)#Ns x M^2

    E_s = N.matmul(ZZ,J)# Ns x Ni
    
    #Para computar el campo total, es el incidente más el disperso
    rho_s = R_obs
    phi_s = theta.T.flatten()
    rho = R_obs
    phi = theta.T.flatten()
    absrho = N.zeros((int(Ns),int(Ni)),dtype = complex)
    for mm,angulo in enumerate(phi_s):
        absrho[:,mm] = (rho**2.+rho_s**2.-2.0*rho*rho_s*N.cos(phi-angulo))**0.5
        
    #Cuidado: ver expresión Ecuación siguiente a Ec. (5-102) en Harrington
    #con I = 1 (fuente de corriente) o pag 60 Pastorino
    E_inc_antenas = -2*pi*freq*mu0/4*special.hankel2(0,kb*absrho)
    
    return E_s+E_inc_antenas

def A(J,Z,M,landa,epsono_r,epsrCb):
    ## for ii = 1:size(J,2);
    ## temp1 = ifft2(fft2(Z).*fft2(reshape(J(:,ii),M,M),2*M-1,2*M-1));
    ## temp2 = temp1(1:M,1:M);
    ## opa1(:,ii) = J(:,ii)+((1i*2*pi/lambda/(120*pi)*(epsono_r(:)-1))).*reshape(temp2,M^2,1);
    ## end
    Ni = J.shape[1]
    J = J.reshape(M,M,int(Ni))
    Z = Z[:,:,N.newaxis]#cuidado! Para que funcione de manera equivalente tile y repmat
    Z = N.tile(Z,(1,1,int(Ni)))
    opa = N.fft.ifft2(N.fft.fft2(Z,axes = (0,1))*N.fft.fft2(J,(2*M-1,2*M-1),(0,1)),axes = (0,1))
    opa = opa[0:M,0:M,:]
    opa = opa.reshape((M**2,int(Ni)))
    opa = J.reshape((M**2,int(Ni)))+(1j*2*pi/landa/(120*pi))*(N.tile(epsono_r.T.flatten().reshape((M**2,1)),(1,int(Ni)))-epsrCb)*opa
    
    return opa

def AH(J,Z,M,landa,epsono_r,epsrCb):
    ## for ii = 1:size(J,2);
    ## temp1 = ifft2(fft2(Z).*fft2(reshape(J(:,ii),M,M),2*M-1,2*M-1));
    ## temp2 = temp1(1:M,1:M);
    ## opa1(:,ii) = J(:,ii)+((1i*2*pi/lambda/(120*pi)*(epsono_r(:)-1))).*reshape(temp2,M^2,1);
    ## end
    Ni = J.shape[1]
    J = J.reshape(M,M,Ni)
    Z = Z[:,:,N.newaxis]#cuidado! Para que funcione de manera equivalente tile y repmat
    Z = N.tile(Z,(1,1,Ni))
    opa = N.fft.ifft2(N.fft.fft2(N.conj(Z),axes = (0,1))*N.fft.fft2(J,(2*M-1,2*M-1),(0,1)),axes = (0,1))
    opa = opa[0:M,0:M,:]
    opa = opa.reshape((M**2,Ni))
    opa = J.reshape((M**2,Ni))+N.conj((1j*2*pi/landa/(120*pi))*(N.tile(epsono_r.T.flatten().reshape((M**2,1)),(1,Ni))-epsrCb))*opa
    
    return opa


def runFem(cilindro, acoplante, trans, receptor, tx, caja):
    """
    Esta función realiza simulaciones de elementos finitos,
    utilizando las bibliotecas DolfinX y Pygmsh para resolver ecuaciones de dispersión
    electromagnética en un cilindro. Puede trabajar con una malla preexistente o
    generar una malla circular si no se proporciona una malla en el objeto `cilindro`.

    Parámetros:
    cilindro (objeto): Objeto que describe el cilindro dispersor. Debe contener la
                      información necesaria sobre la geometría del cilindro.
    acoplante (objeto): Objeto que describe el acoplante utilizado en la simulación.
    trans (objeto): Objeto que describe la antena transmisora utilizada en la simulación.

    Retorna:
    epsilonC (dolfinx.fem.Function): Una función que representa la permitividad
                                    del medio dispersor en la malla simulada.
                                    Si se proporciona una malla preexistente, esta
                                    función contiene la distribución de permitividad
                                    en la malla. Si se genera una malla circular,
                                    la función contiene la distribución de permitividad
                                    en esa malla generada.
    """
    import numpy as np
    from mpi4py import MPI
    import dolfinx
    import ufl
    import sys
    from petsc4py import PETSc
    from scipy import constants as S

    epsilonC = None


    pi = S.pi
    mu0 = S.mu_0
    epsilon0 = S.epsilon_0
    if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
        print("This tutorial requires complex number support")
        sys.exit(0)
    else:
        print(f"Using {PETSc.ScalarType}.")

    freq = 0.4e9
    sigmadc = 0.0
    epsb = 1.0#aire
    epsc = 1.2#cilindro
    sx = caja[0]
    sy = caja[1]
    # ## Defining model parameters
    # wavenumber in coupling media
    kb = 2 * pi * freq * (mu0*epsilon0*epsb)**0.5# 2*pi*f*(mu0*epsilon0*(epsr-jsigma/(2pi*f*eps0)))**0.5
    #wavenumber in cylinder
    kc = 2 * pi * freq * (mu0*epsilon0*epsc)**0.5
    # Corresponding wavelength
    lmbda = 2 * pi / kb.real
    # Polynomial degree
    degree = 6
    # Mesh order
    mesh_order = 2

    if cilindro.thereIsMesh:
        #cargo la malla generada con Gmsh
        mesh, cell_tags, facet_tags = gmshio.read_from_msh(cilindro.mesh, MPI.COMM_WORLD, 0, gdim=mesh_order)
    else:
        import pygmsh

        sx = caja[0]
        sy = caja[1]
        resolution = 0.2
        xc = cilindro.xc
        yc = cilindro.yc

        # Channel parameters
        L = sx #long caja sx
        H = sy #altura (sy)
        c = [xc, yc, 0] #centro del cilindro
        r = cilindro.radio #radio del cilindro

        # Initialize empty geometry using the build in kernel in GMSH
        geometry = pygmsh.geo.Geometry()

        # Fetch model we would like to add data to
        model = geometry.__enter__()
        # Add circle
        circle = model.add_circle(c, r, mesh_size=resolution)
        # Add points
        points = [model.add_point((-L/2, -H/2, 0), mesh_size=resolution),
                  model.add_point((L/2, -H/2, 0), mesh_size=resolution),
                  model.add_point((L/2, H/2, 0), mesh_size=resolution),
                  model.add_point((-L/2, H/2, 0), mesh_size=resolution)]

        # Add lines between all points creating the rectangle
        channel_lines = [model.add_line(points[i], points[i+1])
                         for i in range(-1, len(points)-1)]
        # Create a line loop and plane surface for meshing
        channel_loop = model.add_curve_loop(channel_lines)
        plane_surface = model.add_plane_surface(
            channel_loop, holes=[circle.curve_loop])
        plane_surface2 = model.add_plane_surface(
            circle.curve_loop)
        # Call gmsh kernel before add physical entities
        model.synchronize()
        volume_marker = 6
        model.add_physical([plane_surface], '1')
        model.add_physical([plane_surface2], '2')
        model.add_physical([channel_lines[0], channel_lines[1], channel_lines[2], channel_lines[3]], '10')

        geometry.generate_mesh(dim=2)
        gmsh.write("mesh.msh")
        gmsh.clear()
        geometry.__exit__()
        #cargo la malla generada con pygmsh
        mesh, cell_tags, facet_tags = gmshio.read_from_msh("mesh.msh", MPI.COMM_WORLD, 0, gdim=mesh_order)

    W = dolfinx.fem.FunctionSpace(mesh, ("DG", 0))
    k = dolfinx.fem.Function(W)
    k.x.array[:] = kb
    k.x.array[cell_tags.find(2)] = kc

    epsilonC = dolfinx.fem.Function(W)
    epsilonC.x.array[:] = epsb
    epsilonC.x.array[cell_tags.find(2)] = epsc


    #import matplotlib.pyplot as plt
    #from dolfinx.plot import create_vtk_mesh

    n = ufl.FacetNormal(mesh)
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
    dS = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
    x = ufl.SpatialCoordinate(mesh)
    # ## Variational form
    element = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree)
    V = dolfinx.fem.FunctionSpace(mesh, element)

    #Fuente de corriente en la antena transmisora.
    #https://fenicsproject.discourse.group/t/dirac-delta-distribution-dolfinx/7532/3
    cte = 0.05
    xt = 5.0
    yt = 0.0
    J_a = -1/(2*np.abs(cte*cte)*np.sqrt(pi))*ufl.exp((-((x[0]-xt)/cte)**2-((x[1]-yt)/cte)**2)/2)

    #J = dolfinx.fem.Function(V)
    #dofs = dolfinx.fem.locate_dofs_geometrical(V,  lambda x: np.isclose(x.T, [0.075, 0.0, 0.0]).all(axis=1))
    #J.x.array[dofs] = -1.0

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = - ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx     + k**2 * ufl.inner(u, v) * ufl.dx     - 1j * k * ufl.inner(u, v) * dS
    L = ufl.inner(J_a, v) * ufl.dx

    # Linear solver
    opt = {"ksp_type": "preonly", "pc_type": "lu"}
    problem = dolfinx.fem.petsc.LinearProblem(a, L, petsc_options=opt)
    uh = problem.solve()
    uh.name = "u"

    ## Postprocessing
    from dolfinx.io import XDMFFile #, VTXWriter
    u_abs = dolfinx.fem.Function(V, dtype=np.float64)
    u_abs.x.array[:] = np.abs(uh.x.array)

    # XDMF writes data to mesh nodes
    with XDMFFile(MPI.COMM_WORLD, "out.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(uh)

    with XDMFFile(MPI.COMM_WORLD, "wavenumberNuevo.xdmf", "w") as file:
        file.write_mesh(mesh)
        file.write_function(epsilonC)

    return epsilonC #retorna permitividad





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
    

