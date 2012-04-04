
cimport numpy as np


cdef class NeighborsTable:

    cdef public double r_forcefield_cutoff
    cdef public double r_skin
    cdef unsigned int* neighbor_indices
    cdef size_t N_neighbors, N_allocated

    cdef int _rebuild_neigbors(self, np.ndarray[double, ndim=2] positions, double box_size) except -1

    cdef int add_neighbor(self, unsigned int i, unsigned int j) except -1

    cdef int grow_table(self) except -1


cdef class ForceField:

    # The following three methods operate on positions of particle
    # in a system using the NeighborsTable specified pairs.
    # By default use subsequent pairwise functions and can
    # be overrided in subclasses for efficiency.
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

    cdef int _evaluate_excess_virial(self,
                                     double *,
                                     np.ndarray[double, ndim=2, mode='c'] positions,
                                     double box_size,
                                     NeighborsTable neighbors) except -1

    # Pair wise methods that each dervied ForceField must implement
    # force on particle i (negative force on particle j)
    cdef int _evaluate_a_force(self,
                               double force[3],
                               double r_ij[3]) except -1

    cdef int _evaluate_a_scalar_force(self, double *, double r) except -1

    cdef int _evaluate_a_potential(self, double *, double r) except -1


cdef class LJForceField(ForceField):
    '''6-12 Lennard-Jones ForceField.
       Potential is truncated and shifted to zero at r_cutoff.
    '''

    cdef public double sigma
    cdef public double epsilon
    cdef public double r_cutoff
    cdef public double U_shift


cdef class BasePyForceField(ForceField):
    '''Base class for derived ForceFields written in pure Python
    '''


cdef class BaseConfig:
    '''Base state for a periodic system of particles
    '''
    cdef public:
        np.ndarray positions
        np.ndarray last_positions
        double box_size
        double dt
        double sigma

