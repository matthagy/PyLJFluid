#  Copyright (C) 2012 Matt Hagy <hagy@gatech.edu>
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

cimport util

cimport numpy as np
import numpy as np

def periodic_distance_1D(double a, double b, double box_size):
    return util.c_periodic_distance_1D(a, b, box_size)

def periodic_distances(np.ndarray[double, ndim=2, mode='c'] a not None,
                       np.ndarray[double, ndim=2, mode='c'] b not None,
                       double box_size,
                       np.ndarray[double, ndim=2, mode='c'] out=None):
    if (<object>a).shape != (<object>b).shape:
        raise ValueError("inconsistent shapes %s & %s" % ((<object>a).shape, (<object>b).shape))
    if out is None:
        out = np.empty_like(a)
    cdef double *pa = <double *>np.PyArray_DATA(a)
    cdef double *pb = <double *>np.PyArray_DATA(b)
    cdef double *po = <double *>np.PyArray_DATA(out)
    cdef size_t N = a.size
    cdef size_t i
    with nogil:
        for i in range(N):
            po[i] = util.c_periodic_distance_1D(pa[i], pb[i], box_size)
    return out
