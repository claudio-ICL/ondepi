# distutils: language = c++
# cython: language_level = 3

cimport numpy as np

from libcpp.vector cimport vector
from libcpp.map cimport map as cmap
from ondepi.resources.ingredients cimport (
        HawkesParam, Sample, EventType,
)
from ondepi.resources.intensity.intensity cimport Intensity, IntensityVal
from ondepi.resources.simulation.simulation cimport simulate
from ondepi.resources.filter.filter cimport Z_hat, Z_hat_t

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
    cpdef vector[Z_hat_t] get_filter_process(self) except *
    cpdef vector[double] get_filter_times(self) except *
    cpdef vector[int] get_filter_dD_t(self) except *
    cpdef vector[int] get_filter_dA_t(self) except *
    cpdef void set_price(self, long price) except *
    cpdef void set_param(self,
        double alpha_D_0 ,double alpha_D_1, double alpha_D_2,
        double beta_D, double nu_D,
        double alpha_A_0, double alpha_A_1, double alpha_A_2, 
        double beta_A, double nu_A
    ) except *  
    cpdef void simulate(self, 
            double max_time, long unsigned int max_events,
            EventType first_event,
            long first_state
    ) except *  
    cdef void _filter(self, double dt, long unsigned int num_states)
    cpdef void filter(self, double dt, long unsigned int num_states) except *
    cpdef void calibrate(self, 
            Sample sample, 
            int maxiter=*, 
            float xtol=*, 
            int disp=*) except *
