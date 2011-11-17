
import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, realloc, free

from util cimport c_periodic_distance


cdef class Parameters:

    def __init__(self, mass=1.0, delta_t=0.01):
        self.mass = mass
        self.delta_t = delta_t


cdef class NeighborsTable:

    def __cinit__(self, r_neighbor):
        self.r_neighbor = r_neighbor
        self.neighbor_indices = NULL
        self.N_neighbors = 0
        self.N_allocated = 0

    def __dealloc__(self):
        free(self.neighbor_indices)

    cdef _rebuild_neigbors(self, np.ndarray[double, ndim=2] positions):
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

    def __cinit__(self, sigma=1.0, epsilon=1.0, r_cutoff=1.0):
        self.sigma = sigma
        self.epsilon = epsilon
        self.r_cutoff = r_cutoff


cdef class Config:

    pass

cdef class System:

    pass
