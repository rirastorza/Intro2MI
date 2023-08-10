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
xt = 6.0
yt = 0.0
J_a = -2e3/(2*np.abs(cte*cte)*np.sqrt(pi))*ufl.exp((-((x[0]-xt)/cte)**2-((x[1]-yt)/cte)**2)/2)

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
print(type(uh))
#np.savez('Ez_salidas.npz',ezr = uh.x.array.real,ezi = uh.x.array.imag)

# XDMF writes data to mesh nodes
with XDMFFile(MPI.COMM_WORLD, "out.xdmf", "w") as file:
    file.write_mesh(mesh)
    file.write_function(u_abs)


nant_r = 16 #antenas receptoras
rant_r = 3 #radio de antenas receptoras[m]
#Coordenadas antenas receptoras
angulo_r = np.linspace(0.0, 2.0*pi-2*pi/nant_r, nant_r)
print(len(angulo_r))
xantenas_r= (rant_r)*np.cos(angulo_r)
yantenas_r = (rant_r)*np.sin(angulo_r)



#Evaluar puntos
points = np.zeros((3, len(xantenas_r)))
points[0] = xantenas_r
points[1] = yantenas_r
u_values = []

from dolfinx import geometry
bb_tree = geometry.BoundingBoxTree(mesh, mesh.topology.dim)

cells = []
points_on_proc = []
# Find cells whose bounding-box collide with the the points
cell_candidates = geometry.compute_collisions(bb_tree, points.T)
# Choose one of the cells that contains the point
colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, points.T)
for i, point in enumerate(points.T):
    if len(colliding_cells.links(i))>0:
        points_on_proc.append(point)
        cells.append(colliding_cells.links(i)[0])

points_on_proc = np.array(points_on_proc, dtype=np.float64)
u_values = uh.eval(points_on_proc, cells)

print(u_values,type(u_values))

import matplotlib.pyplot as plt
fig, (axs1,axs2) = plt.subplots(2,1,figsize=(16, 10))
N=9
cmap = plt.get_cmap('Set2', N)




axs1.plot(abs(u_values),'^-', label='módulo FEM',markersize=10,)#color =cmap(0)) 
axs2.plot(-np.angle(u_values),'v-', label='fase FEM',markersize=10,)#color=cmap(2)) 

np.savez('Ez_fem',absEz = abs(u_values),angleEz = -np.angle(u_values))

#for n in range(0,len(xantenas_r)):
    #print(uh(xantenas_r[n],yantenas_r[n]))

#with XDMFFile(MPI.COMM_WORLD, "wavenumber.xdmf", "w") as file:
    #file.write_mesh(mesh)
    #file.write_function(k)

##from matplotlib import pyplot as plt

##plt.figure(1)
##extent2=[-0.25/2,0.25/2,-0.25/2,0.25/2]
##plt.imshow(u_abs.x.array[:].transpose(),extent = extent2)
plt.show()

