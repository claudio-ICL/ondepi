# distutils: language = c++
# cython: language_level = 3

from libcpp.vector cimport vector
cimport numpy as np

cdef enum EventType:
    D, A

cdef struct EventState:
    double time
    EventType event
    long state

cdef struct Sample:
    vector[EventState] observations

cdef struct HawkesParam:
    double alpha_0
    double alpha_1
    double alpha_2
    double beta
    double nu

cdef struct EvalLoglikelihood:
    double logL
    vector[double] gradient

#cdef vector[double] times_D_in_sample(Sample sample)    
#cdef vector[long] states_D_in_sample(Sample sample)    

cdef class Process:
    cdef Sample* sample
    cdef vector[double] times
    cdef void set_sample(self, Sample* sample)
    cdef void init_times(self, double dt)
    cpdef void p_init_times(self, double dt)
    cdef vector[double] get_times(self)
    cdef void set_times(self, vector[double] times)
    cdef vector[int] dD_t
    cdef vector[int] get_dD_t(self)
    cdef void set_dD_t(self, vector[int] dD_t)
    cdef vector[int] dA_t
    cdef vector[int] get_dA_t(self)
    cdef void set_dA_t(self, vector[int] dA_t)
