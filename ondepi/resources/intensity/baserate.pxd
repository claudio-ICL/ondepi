# distutils: language = c++
# cython: language_level = 3

from libc.math cimport exp
from libc.math cimport log

cdef double phi(double x)
cdef double phi_prime(double x)

cdef double nu(double q, double c_0, double c_1, double c_2)
cdef double nu_partial_0(double q, double c_0, double c_1, double c_2)
cdef double nu_partial_1(double q, double c_0, double c_1, double c_2)
cdef double nu_partial_2(double q, double c_0, double c_1, double c_2)


cdef class BaseRate:
    cdef double c_0
    cdef double c_1
    cdef double c_2
    cdef long state
    cdef double value
    cdef double compute_at_arrival(self, long new_state)
    cdef void set_state(self, long q)
    cdef double get_value(self)
    cdef double eval_(self, double q)
    cdef double eval__(self, long q)
    cdef double partial_0(self, double q)
    cdef double partial_0_(self, long q)
    cdef double get_partial_0(self)
    cdef double partial_1(self, double q)
    cdef double partial_1_(self, long q)
    cdef double get_partial_1(self)
    cdef double partial_2(self, double q)
    cdef double partial_2_(self, long q)
    cdef double get_partial_2(self)

