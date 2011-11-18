
cimport numpy as np


cdef class Parameters:

    cdef public double mass
    cdef public double delta_t


cdef class NeighborsTable:

    cdef public double r_forcefield_cutoff
    cdef public double r_skin
    cdef int* neighbor_indices
    cdef int N_neighbors, N_allocated

    cdef _rebuild_neigbors(self, np.ndarray[double, ndim=2] positions)


cdef class ForceField:

    cdef void _evaluate_force(self,
                              np.ndarray[double, ndim=2] forces,
                              np.ndarray[double, ndim=2] positions,
                              NeighborsTable neighbors)

    cdef double _evaluate_potential(self,
                                    np.ndarray[double, ndim=2] positions,
                                    NeighborsTable neighbors)

    # force on particle i (negative force on particle j)
    cdef void _evalaute_a_force(self,
                                double force[3],
                                double pos_i[3],
                                double pos_j[3])

    cdef double _evaluate_a_scalar_force(self, double pos_i, double pos_j)

    cdef double _evaluate_a_scalar_potential(self, double pos_i, double pos_j)


cdef class LJForceFeild(ForceField):

    cdef public double sigma
    cdef public double epsilon
    cdef public double r_cutoff


cdef class Config:

    cdef public object positions
    cdef public object last_positions
    cdef public double box_size

cdef class System:

    cdef public Parameters parameters
    cdef public Config current_config
