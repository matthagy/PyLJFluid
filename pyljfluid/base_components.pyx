
from __future__ import division

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, realloc, free

from util cimport c_periodic_distance


cdef class Parameters:

    def __init__(self, mass=1.0, delta_t=0.01):
        self.mass = mass
        self.delta_t = delta_t

cdef class NeighborsTable:

    def __cinit__(self, double r_forcefield_cutoff, double r_skin):
        self.r_forcefield_cutoff = r_forcefield_cutoff
        self.r_skin = r_skin
        self.neighbor_indices = NULL
        self.N_neighbors = 0
        self.N_allocated = 0

    def __dealloc__(self):
        free(self.neighbor_indices)

    cdef _rebuild_neigbors(self, np.ndarray[double, ndim=2] positions):
        pass

    def rebuild_neighbors(self, np.ndarray[double, ndim=2] positions):
        pass


cdef class ForceField:

    cdef void _evaluate_force(self,
                              np.ndarray[double, ndim=2] forces,
                              np.ndarray[double, ndim=2] positions,
                              NeighborsTable neighbors):
        pass

    cdef double _evaluate_potential(self,
                                    np.ndarray[double, ndim=2] positions,
                                    NeighborsTable neighbors):
        pass

    cdef void _evalaute_a_force(self,
                                double force[3],
                                double pos_i[3],
                                double pos_j[3]):
        pass

    cdef double _evaluate_a_scalar_force(self, double pos_i, double pos_j):
        pass

    cdef double _evaluate_a_scalar_potential(self, double pos_i, double pos_j):
        pass


cdef class LJForceFeild(ForceField):

    def __cinit__(self, sigma=1.0, epsilon=1.0, r_cutoff=2.5):
        self.sigma = sigma
        self.epsilon = epsilon
        self.r_cutoff = r_cutoff


cdef class BaseConfig:

    def __cinit__(self,
                  np.ndarray[double, ndim=2] positions,
                  np.ndarray[double, ndim=2] last_positions,
                  double box_size):
        self.positions = positions
        self.last_positions = last_positions
        self.box_size = box_size

    @classmethod
    def create(cls, N, rho, sigma=1.0, T=1.0, mass=1.0):
        V = N * sigma**3 / rho
        box_size = V**(1/3)
        positions = np.random.uniform(0.0, box_size, (N, 3))
        velocities = np.random.normal(scale=np.sqrt(T / mass))
        return cls(positions, positions - velocities, box_size)

    def calculate_velocities(self):
        return self.positions - self.last_positions

    def calculate_rms_velocity(self):
        v = self.calculate_temperature()
        (v**2).mean()**0.5

    def calculate_temperature(self):
        v_rms = self.calculate_rms_velocity()


cdef class System:

    pass
