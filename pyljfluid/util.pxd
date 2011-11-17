# declaration of Cython utilities

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

cdef inline void c_periodic_distance(double lv[3], double av[3], double bv[3], double box_size) nogil:
    """Calculated periodic distance along three dimensions
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

