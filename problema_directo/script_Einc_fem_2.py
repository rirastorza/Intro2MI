#!/usr/bin/env python3
# coding: utf-8

#Script para problema directo de microondas con FENICSx

#IMPORTANTE: antes de simular correr en consola lo siguiente.
#import os
#string = "export PETSC_DIR=/usr/lib/petscdir/petsc3.15/x86_64-linux-gnu-complex"
#os.system(string)


import numpy as np
from mpi4py import MPI
import dolfinx
import ufl
import sys
from petsc4py import PETSc
from scipy import constants as S
from dolfinx.io import gmshio

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
sx = 14.0
sy = 14.0
# ## Defining model parameters
# wavenumber in coupling media
kb = 2 * pi * freq * (mu0*epsilon0*epsb)**0.5
#wavenumber in cylinder
kc = 2 * pi * freq * (mu0*epsilon0*epsc)**0.5
# Corresponding wavelength
lmbda = 2 * pi / kb.real
# Polynomial degree
degree = 6
# Mesh order
mesh_order = 2

#cargo la malla generada con Gmsh
mesh, cell_tags, facet_tags = gmshio.read_from_msh("modelo_prueba.msh", MPI.COMM_WORLD, 0, gdim=mesh_order)

#import gmsh
#gmsh.initialize()
#caja = gmsh.model.occ.addRectangle(-sx/2, -sy/2, sx/2, sy, sx)
#disco = gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
#diferencia = gmsh.model.occ.cut([(2, caja)], [(2, disco)])
#gmsh.model.occ.synchronize()

#gmsh.model.mesh.generate(2)
#gmsh.write("mesh.msh")
#gmsh.finalize()

#mesh = dolfinx.mesh.create_rectangle(comm=MPI.COMM_WORLD,
                            #points=((-0.125, -0.125), (0.125, 0.125)), n=(50, 50),
                            #cell_type=dolfinx.mesh.CellType.triangle,)


##mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 50, 50)

W = dolfinx.fem.FunctionSpace(mesh, ("DG", 0))
k = dolfinx.fem.Function(W)
k.x.array[:] = kb
k.x.array[cell_tags.find(2)] = kc
#import matplotlib.pyplot as plt
#from dolfinx.plot import create_vtk_mesh

n = ufl.FacetNormal(mesh)
dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)
dS = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
x = ufl.SpatialCoordinate(mesh)
##uinc = ufl.exp(1j * k * x[0])
##g = ufl.dot(ufl.grad(uinc), n) - 1j * k * uinc


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
    file.write_function(u_abs)

with XDMFFile(MPI.COMM_WORLD, "wavenumber.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(k)

##from matplotlib import pyplot as plt

##plt.figure(1)
##extent2=[-0.25/2,0.25/2,-0.25/2,0.25/2]
##plt.imshow(u_abs.x.array[:].transpose(),extent = extent2)
##plt.show()

