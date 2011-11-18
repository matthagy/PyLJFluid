
from __future__ import division

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, realloc, free

from util cimport c_periodic_distance

def ensure_positions_array(arr):
    if not isinstance(arr, np.ndarray):
        raise TypeError("bad argument for positions_array of type %s; must be an ndarray"
                        % (type(arr).__name__,))
    if arr.dtype != np.double:
        raise ValueError("bad array type %s; must be double" % (arr.dtype,))
    if not len(arr.shape) == 2 and arr.shape[1] == 3:
        raise ValueError("bad array shape %s; must be (N,3)" % (arr.shape,))
    return arr

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

    cdef inline int add_neighbor(self, unsigned int i, unsigned int j) except -1:
        if self.N_neighbors == self.N_allocated:
            self.grow_table()

        self.neighbor_indices[2 * self.N_neighbors] = i
        self.neighbor_indices[2 * self.N_neighbors + 1] = j
        self.N_neighbors += 1

    property size:
        def __get__(self):
            return self.N_neighbors

    property allocated_size:
        def __get__(self):
            return self.N_allocated

    cdef inline int grow_table(self) except -1:
        cdef size_t alloc_size = 8192
        if self.neighbor_indices == NULL:
            self.N_allocated = alloc_size
            self.neighbor_indices = <unsigned int*>malloc(self.N_allocated * 2 * sizeof(unsigned int))
        else:
            self.N_allocated += alloc_size
            self.neighbor_indices = <unsigned int*>realloc(self.neighbor_indices, self.N_allocated * 2 * sizeof(unsigned int))
        if self.neighbor_indices == NULL:
            raise MemoryError
        assert self.N_allocated > self.N_neighbors

    cdef int _rebuild_neigbors(self, np.ndarray[double, ndim=2] positions, double box_size) except -1:
        ensure_positions_array(positions)
        cdef int i, j, k, N
        cdef double r_sqr, r_neighbor_sqr
        cdef double l, half_size
        N = positions.shape[0]

        self.N_neighbors = 0
        r_neighbor_sqr = (self.r_skin + self.r_forcefield_cutoff) ** 2
        half_size = 0.5 * box_size

        for i in range(N):
            for j in range(i+1, N):

                r_sqr = 0.0
                for k in range(3):
                    l = positions[i, k] - positions[j, k]
                    if l > half_size:
                        l -= box_size
                    elif l < -half_size:
                        l += box_size
                    r_sqr += l*l

                if r_sqr < r_neighbor_sqr:
                    self.add_neighbor(i, j)

    def rebuild_neighbors(self, op, box_size=None):
        if isinstance(op, BaseConfig):
            assert box_size is None
            positions = op.positions
            box_size = op.box_size
        else:
            positions = op
            assert box_size is not None
        self._rebuild_neigbors(positions, box_size)


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
