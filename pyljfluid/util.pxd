# declaration of Cython utilities

from libc.math cimport sqrt
cimport cython

cdef inline double c_periodic_distance_1D(double a, double b, double box_size) nogil:
    """Calculated periodic distance along a single dimension
    """
    cdef double l = b - a
    cdef double h = 0.5 * box_size
    if l > h:
        l -= box_size
    elif l < -h:
        l += box_size
    return l

cdef inline void c_periodic_direction(double lv[3], double av[3], double bv[3], double box_size) nogil:
    """Calculated periodic direction along three dimensions
    """
    cdef double h = 0.5 * box_size
    cdef double l
    for i in xrange(3):
        l = bv[i] - av[i]
        if l > h:
            l -= box_size
        elif l < -h:
            l += box_size
        lv[i] = l


cdef inline double c_periodic_distance_sqr(double av[3], double bv[3], double box_size) nogil:
    """Calculated square length of the periodic direction along three dimensions
    """
    cdef double h = 0.5 * box_size
    cdef double acc=0.0
    cdef double l
    for i in xrange(3):
        l = bv[i] - av[i]
        if l > h:
            l -= box_size
        elif l < -h:
            l += box_size
        acc += l*l
    return acc

cdef inline double c_periodic_distance(double av[3], double bv[3], double box_size) nogil:
    return sqrt(c_periodic_distance_sqr(av, bv, box_size))

cdef inline double c_vector_sqr_length(double v[3]) nogil:
    return v[0]*v[0] + v[1]*v[1] + v[2]*v[2]

cdef inline double c_vector_length(double v[3]) nogil:
    return sqrt(c_vector_sqr_length(v))

@cython.cdivision(True)
cdef inline signed int wrapping_modulo(signed int dividend, signed int divisor) nogil:
    cdef signed int mod = cython.cmod(dividend, divisor)
    if mod < 0:
        return divisor + mod
    return mod

assert wrapping_modulo(10, 3) == 10 % 3
assert wrapping_modulo(-10, 3) == -10 % 3
