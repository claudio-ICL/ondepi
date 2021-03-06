# distutils: language = c++
# cython: language_level = 3

cimport numpy as np

from libc.math cimport log10, ceil
from libcpp.vector cimport vector
from libcpp.map cimport map as cmap
from ondepi.resources.ingredients cimport (
        HawkesParam, Sample, EventType,
)
from ondepi.resources.intensity.intensity cimport Intensity, IntensityVal
from ondepi.resources.simulation.simulation cimport simulate
from ondepi.resources.filter.filter cimport Z_hat, Z_hat_t, regularise_expected_values

ctypedef cmap[EventType, HawkesParam] QueueParam    

cdef class Queue:
    cdef long price
    cdef QueueParam param
    cdef Sample sample
    cpdef void set_sample(self, Sample sample) except *  
    cpdef Sample get_sample(self) except *  
    cdef Intensity intensity
    cdef Z_hat z_hat
    cpdef vector[IntensityVal] get_intensity_process(self) except *
    cpdef vector[double] get_intensity_times(self) except *
    cpdef void populate_intensity(self, double dt)
    cpdef vector[Z_hat_t] get_filter_process(self) except *
    cpdef vector[double] get_filter_times(self) except *
    cpdef vector[int] get_filter_dD_t(self) except *
    cpdef vector[int] get_filter_dA_t(self) except *
    cpdef void set_price(self, long price) except *
    cpdef long get_price(self) except *
    cpdef void _set_param(self,
        double nu_D_0 ,double nu_D_1, double nu_D_2,
        double alpha_D, double beta_D,
        double nu_A_0, double nu_A_1, double nu_A_2, 
        double alpha_A, double beta_A
    ) except *  
    cpdef QueueParam get_param(self)
    cpdef void simulate(self, 
            double max_time, long unsigned int max_events,
            EventType first_event,
            long first_state
    ) except *  
    cdef void _filter(self, double dt, long unsigned int num_states)
    cpdef void filter(self, double dt, long unsigned int num_states) except *
    cpdef void calibrate_on_self(self, 
            int num_guesses=*,
            double ftol=*,
            double gtol=*,
            int maxiter=*, 
            int disp=*,
            launch_async=*, 
            ) except *
    cpdef void calibrate(self, 
            Sample sample, 
            int num_guesses=*,
            double ftol=*,
            double gtol=*,
            int maxiter=*, 
            int disp=*,
            launch_async=*, 
            ) except *
