# distutils: language = c++
# cython: language_level = 3

cimport numpy as np
from libc.math cimport exp, log
from libcpp.vector cimport vector
from libcpp.map cimport map as cmap
from ondepi.resources.ingredients cimport EventType, Process
from ondepi.resources.intensity.intensity cimport (
        Intensity, IntensityVal, 
        IntensityValForFilterUpdate
)

cdef enum Neighbours:
     _below
     _same
     _above

ctypedef cmap[Neighbours, double] Z_hat_t_local

cdef struct Z_hat_t:
    double time
    double expected_value
    vector[double] distribution

cdef double update_0(double z_hat_t_0, double z_hat_t_1, 
    IntensityValForFilterUpdate intensity,
    double mu,
    int dD_t,
    double dt
)

cdef double update_local(
    Z_hat_t_local z_hat_t,
    IntensityValForFilterUpdate intensity,
    double mu,
    int dD_t,
    double dt
) 

cdef class Z_hat(Process):
    cdef vector[Z_hat_t] process
    cdef void init_process(self)
    cdef vector[Z_hat_t] get_process(self)
    cdef vector[IntensityVal] intensities
    cdef Intensity intensity
    cdef void set_intensity(self, Intensity intensity)
    cdef Intensity get_intensity(self)
    cdef Z_hat_t get_slice(self, long unsigned int idx)
    cdef Z_hat_t get_time_slice(self, double t)
    cdef void populate(self, long unsigned int num_states)

cpdef np.ndarray[double, ndim=1] regularise_expected_values(
        np.ndarray[double, ndim=1] times, 
        np.ndarray[long, ndim=1] states, 
        np.ndarray[double, ndim=1] z_hat, 
        double beta)


cdef vector[double] _regularise_expected_values(
        vector[double] times, 
        vector[long] states,
        vector[double] expected_values,
        double beta)
