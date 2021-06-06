# distutils: language = c++
# cython: language_level = 3

from libcpp.vector cimport vector
from libcpp.map cimport map as cmap
from ondepi.resources.ingredients cimport EventType, Process
from ondepi.resources.intensity.intensity cimport Intensity, IntensityVal

cdef enum Neighbours:
     _below
     _same
     _above

ctypedef cmap[Neighbours, double] Z_hat_t_local

cdef struct Z_hat_t:
    double time
    double expected_value
    vector[double] distribution

cdef double update_0(double z_hat_t_0, double z_hat_t_1, IntensityVal ival, int dD_t, double dt)

cdef double update_local(
    Z_hat_t_local z_hat_t,
    double z_hat_t_0,
    IntensityVal ival,
    int dD_t,
    double dt
) 

cdef class Z_hat(Process):
    cdef vector[Z_hat_t] process
    cdef void init_process(self)
    cdef vector[Z_hat_t] get_process(self)
    cdef vector[double] get_expected_process(self)
    cdef vector[IntensityVal] intensities
    cdef void set_intensities(self, vector[IntensityVal] intensities)
    cdef vector[IntensityVal] get_intensities(self)
    cdef Z_hat_t get_slice(self, long unsigned int idx)
    cdef Z_hat_t get_time_slice(self, double t)
    cdef void populate(self, long unsigned int num_states)

