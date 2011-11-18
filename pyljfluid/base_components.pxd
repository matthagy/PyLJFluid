
cimport numpy as np


cdef class Parameters:

    cdef public double mass
    cdef public double delta_t


cdef class NeighborsTable:

    cdef public double r_forcefield_cutoff
    cdef public double r_skin
    cdef unsigned int* neighbor_indices
    cdef size_t N_neighbors, N_allocated

    cdef int _rebuild_neigbors(self, np.ndarray[double, ndim=2] positions, double box_size) except -1

    cdef inline int add_neighbor(self, unsigned int i, unsigned int j) except -1

    cdef inline int grow_table(self) except -1


cdef class ForceField:

    cdef int _evaluate_forces(self,
                              np.ndarray[double, ndim=2, mode='c'] forces,
                              np.ndarray[double, ndim=2, mode='c'] positions,
                              double box_size,
                              NeighborsTable neighbors) except -1

    cdef int _evaluate_potential(self,
                                 double *,
                                 np.ndarray[double, ndim=2] positions,
                                 double box_size,
                                 NeighborsTable neighbors) except -1

    # force on particle i (negative force on particle j)
    cdef int _evaluate_a_force(self,
                               double force[3],
                               double r_ij[3]) except -1

    cdef int _evaluate_a_scalar_force(self, double *, double r) except -1

    cdef int _evaluate_a_potential(self, double *, double r) except -1


cdef class LJForceFeild(ForceField):

    cdef public double sigma
    cdef public double epsilon
    cdef public double r_cutoff
    cdef public double U_shift


cdef class BasePyForceField(ForceField):

    pass

cdef class BaseConfig:

    cdef public object positions
    cdef public object last_positions
    cdef public double box_size
    cdef public double dt

cdef class System:

    cdef public Parameters parameters
    cdef public BaseConfig current_config
