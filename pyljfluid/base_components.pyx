
from __future__ import division

import numpy as np
cimport numpy as np

cimport cython
from libc.stdlib cimport malloc, realloc, free

from util cimport c_periodic_direction, c_periodic_distance, c_vector_length


cdef ensure_N3_array(arr):
    if not isinstance(arr, np.ndarray):
        raise TypeError("bad argument of type %s; must be an ndarray"
                        % (type(arr).__name__,))
    if arr.dtype != np.double:
        raise ValueError("bad array type %s; must be double" % (arr.dtype,))
    if not (len(arr.shape) == 2 and arr.shape[1] == 3):
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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef int _rebuild_neigbors(self, np.ndarray[double, ndim=2] positions, double box_size) except -1:
        ensure_N3_array(positions)
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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef int _evaluate_forces(self,
                              np.ndarray[double, ndim=2] forces,
                              np.ndarray[double, ndim=2] positions,
                              double box_size,
                              NeighborsTable neighbors) except -1:
        cdef unsigned int* neighbor_indices = neighbors.neighbor_indices
        cdef size_t N_neighbors = neighbors.N_neighbors
        cdef unsigned int neighbor_i, inx_i, inx_j, k
        cdef double force[3], pos_i[3], pos_j[3], r_ij[3]

        forces.fill(0.0)

        for neighbor_i in range(N_neighbors):
            inx_i = neighbor_indices[2 * neighbor_i]
            inx_j = neighbor_indices[2 * neighbor_i + 1]

            for k in range(3): pos_i[k] = positions[inx_i, k]
            for k in range(3): pos_j[k] = positions[inx_j, k]
            c_periodic_direction(r_ij, pos_i, pos_j, box_size)

            self._evaluate_a_force(force, r_ij)
            for k in range(3): forces[inx_i, k] += force[k]
            for k in range(3): forces[inx_j, k] -= force[k]

    cdef int _evaluate_potential(self, double *U_p,
                                 np.ndarray[double, ndim=2] positions,
                                 double box_size,
                                 NeighborsTable neighbors) except -1:

        cdef unsigned int* neighbor_indices = neighbors.neighbor_indices
        cdef size_t N_neighbors = neighbors.N_neighbors
        cdef unsigned int neighbor_i, inx_i, inx_j, k
        cdef double pos_i[3], pos_j[3], r, U, acc_U

        acc_U = 0.0
        for neighbor_i in range(N_neighbors):
            inx_i = neighbor_indices[2 * neighbor_i]
            inx_j = neighbor_indices[2 * neighbor_i + 1]

            for k in range(3): pos_i[k] = positions[inx_i, k]
            for k in range(3): pos_j[k] = positions[inx_j, k]
            r = c_periodic_distance(pos_i, pos_j, box_size)
            self._evaluate_a_potential(&U, r)
            acc_U += U

        U_p[0] = acc_U


    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef int _evaluate_a_force(self,
                               double force[3],
                               double r_ij[3]) except -1:
        cdef double f, r, scale
        cdef int k
        r = c_vector_length(r_ij)
        self._evaluate_a_scalar_force(&f, r)
        scale = f / r
        for k in range(3): force[k] = r_ij[k] * scale

    cdef int _evaluate_a_scalar_force(self, double *f_ptr, double r) except -1:
        raise RuntimeError("_evaluate_a_scalar_force not overrided in base class")

    cdef int _evaluate_a_potential(self, double *U_ptr, double r) except -1:
        raise RuntimeError("_evaluate_a_potential not overrided in base class")

    def evaluate_forces(self, forces, positions, double box_size, NeighborsTable neighbors):
        ensure_N3_array(forces)
        ensure_N3_array(positions)
        self._evaluate_forces(forces, positions, box_size, neighbors)
        return forces

    def evaluate_potential(self, positions, double box_size, NeighborsTable neighbors):
        ensure_N3_array(positions)
        cdef double U
        self._evaluate_potential(&U, positions, box_size, neighbors)
        return U




@cython.cdivision(True)
cdef inline double evaluate_U612(double sigma, double epsilon, double r) nogil:
    cdef double x = sigma / r
    cdef double x2 = x*x
    cdef double x6 = x2*x2*x2
    return 4 * epsilon * (x6*x6 - x6)

cdef class LJForceFeild(ForceField):

    def __cinit__(self, sigma=1.0, epsilon=1.0, r_cutoff=2.5):
        self.sigma = sigma
        self.epsilon = epsilon
        self.r_cutoff = r_cutoff
        self.U_shift = evaluate_U612(self.sigma, self.epsilon, self.r_cutoff)

    @cython.cdivision(True)
    cdef int _evaluate_a_scalar_force(self, double *f_ptr, double r) except -1:
        if r >= self.r_cutoff:
            f_ptr[0] = 0.0
            return 0

        cdef double inv_r = 1.0 / r
        cdef double x = self.sigma * inv_r
        cdef double x2 = x*x
        cdef double x6 = x2*x2*x2
        f_ptr[0] = -4 * 6 * self.epsilon * inv_r * (2*x6*x6 - x6)

    @cython.cdivision(True)
    cdef int _evaluate_a_potential(self, double *U_ptr, double r) except -1:
        if r >= self.r_cutoff:
            U_ptr[0] = 0.0
            return 0

        U_ptr[0] = evaluate_U612(self.sigma, self.epsilon, r) - self.U_shift


cdef class BasePyForceField(ForceField):
    '''Allows Python derived classes to implement force field
       functionality
    '''

    pass

#     cdef _evaluate_forces(self, forces, positions, neighbors):
#         try:
#             ev = self.evaluate_force
#         except AttributeError:
#             super(BasePyForceField, self)._evaluate_forces(forces, positions, neighbors)
#         else:
#             ev(forces, positions, neighbors)

#     cdef _evaluate_potential(self, positions, neighbors):
#         try:
#             ev = self.evaluate_potential
#         except AttributeError:
#             super(BasePyForceField, self)._evaluate_potential(positions, neighbors)
#         else:
#             ev(positions, neighbors)




cdef class BaseConfig:

    def __cinit__(self,
                  np.ndarray[double, ndim=2] positions,
                  np.ndarray[double, ndim=2] last_positions,
                  double box_size):
        self.positions = positions
        self.last_positions = last_positions
        self.box_size = box_size



cdef class System:

    pass


