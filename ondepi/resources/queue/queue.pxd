# distutils: language = c++
# cython: language_level = 3

from libcpp.map cimport map as cmap
from ondepi.resources.ingredients cimport HawkesParam, Sample, EventType
from ondepi.resources.intensity.intensity cimport Intensity
from ondepi.resources.simulation.simulation cimport simulate
# from ondepi.resources.likelihood.likelihood  cimport EvalLoglikelihood, eval_loglikelihood

ctypedef cmap[EventType, HawkesParam] QueueParam    

cdef class Queue:
    cdef QueueParam param
    cdef Sample sample
    cpdef Sample get_sample(self) except *  
    cpdef void set_sample(self, Sample sample) except *  
    cdef Intensity intensity
    cpdef void set_sample(self, Sample sample) except *  
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
    cpdef void calibrate(self, Sample sample) except *  
