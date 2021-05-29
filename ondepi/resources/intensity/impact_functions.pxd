# distutils: language = c++
# cython: language_level = 3

from libc.math cimport exp
from libc.math cimport log

cdef double phi(double x)
cdef double phi_prime(double x)

cdef double alpha_D_(double q, double c_0, double c_1, double c_2)
cdef double alpha_D_partial_0(double q, double c_0, double c_1, double c_2)
cdef double alpha_D_partial_1(double q, double c_0, double c_1, double c_2)
cdef double alpha_D_partial_2(double q, double c_0, double c_1, double c_2)


cdef class Alpha_D:
    cdef double c_0
    cdef double c_1
    cdef double c_2
    cdef double eval_(self, double q)
    cdef double eval__(self, long q)
    cdef double partial_0(self, double q)
    cdef double partial_0_(self, long q)
    cdef double partial_1(self, double q)
    cdef double partial_1_(self, long q)
    cdef double partial_2(self, double q)
    cdef double partial_2_(self, long q)

cdef double alpha_A_(double q, double c_0, double c_1, double c_2)
cdef double alpha_A_partial_0(double q, double c_0, double c_1, double c_2)
cdef double alpha_A_partial_1(double q, double c_0, double c_1, double c_2)
cdef double alpha_A_partial_2(double q, double c_0, double c_1, double c_2)

cdef class Alpha_A:
    cdef double c_0
    cdef double c_1
    cdef double c_2
    cdef double eval_(self, double q)
    cdef double eval__(self, long q)
    cdef double partial_0(self, double q)
    cdef double partial_0_(self, long q)
    cdef double partial_1(self, double q)
    cdef double partial_1_(self, long q)
    cdef double partial_2(self, double q)
    cdef double partial_2_(self, long q)
  

cdef double next_v(double v, double T, double next_T, double next_alpha,
              double beta)
cdef double next_v_beta(double v_beta, double v, double T, double next_T,
                   double beta)
cdef double next_v_alpha(double v_alpha, double T, double next_T,
                    double next_alpha_partial, double beta)
