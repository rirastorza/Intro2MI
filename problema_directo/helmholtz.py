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

if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
    print("This tutorial requires complex number support")
    sys.exit(0)
else:
    print(f"Using {PETSc.ScalarType}.")


# ## Defining model parameters
# wavenumber in free space (air)
k0 = 10 * np.pi
# Corresponding wavelength
lmbda = 2 * np.pi / k0
# Polynomial degree
degree = 6
# Mesh order
mesh_order = 2

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)

W = dolfinx.fem.FunctionSpace(mesh, ("DG", 0))
k = dolfinx.fem.Function(W)
k.x.array[:] = k0
#k.x.array[cell_tags.find(1)] = 3 * k0

#import pyvista
import matplotlib.pyplot as plt
from dolfinx.plot import create_vtk_mesh

n = ufl.FacetNormal(mesh)
x = ufl.SpatialCoordinate(mesh)
uinc = ufl.exp(1j * k * x[0])
g = ufl.dot(ufl.grad(uinc), n) - 1j * k * uinc

# ## Variational form
element = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), degree)
V = dolfinx.fem.FunctionSpace(mesh, element)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = - ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx     + k**2 * ufl.inner(u, v) * ufl.dx     - 1j * k * ufl.inner(u, v) * ufl.ds
L = ufl.inner(g, v) * ufl.ds


# ## Linear solver
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
