# distutils: language = c++
# cython: language_level = 3

from libc.math cimport exp
from libc.math cimport log

cdef double compute_decay(double time, double previous_time, double beta)

cdef double compute_next_v(double v, double T, double next_T, double beta)

cdef double compute_next_v_beta(double v, double v_beta, double T, double next_T, double beta)

cdef struct ImpactRateValue:
    double time
    double value

cdef class ImpactRate:
    cdef double alpha
    cdef double beta
    cdef ImpactRateValue value 
    cdef double next_value(self, double next_T)
    cdef void innovate(self, double next_T)
    cdef double compute_at_arrival(self, double next_T)
    cdef ImpactRateValue get_value(self)
    cdef double eval_after_last_event(self, double time)

