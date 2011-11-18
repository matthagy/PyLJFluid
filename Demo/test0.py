from __future__ import division

import numpy as np

from pyljfluid.components import (Parameters, NeighborsTable,
                                  LJForceFeild, Config,
                                  System)



parameters = Parameters(
    mass=1.0,
    delta_t=1.0)

forcefield = LJForceFeild(
    sigma=1.0,
    epsilon=1.0,
    r_cutoff=3.3)

neighbors_table = NeighborsTable(
    r_forcefield_cutoff=forcefield.r_cutoff,
    r_skin=1.0)

N_particles = 1000
rho = 0.4
T = 1.35

config = Config.create(N=N_particles, rho=rho, sigma=forcefield.sigma, T=T, mass=parameters.mass)

config.calculate_temperature(mass=parameters.mass)
