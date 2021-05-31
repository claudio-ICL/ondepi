# distutils: language = c++
# cython: language_level = 3

from libcpp.vector cimport vector

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

cdef class Process:
    cdef Sample* sample
    cdef vector[double] times
    cdef vector[int] dD_t
    cdef void set_sample(self, Sample* sample)
    cdef void init_times(self, double dt)
