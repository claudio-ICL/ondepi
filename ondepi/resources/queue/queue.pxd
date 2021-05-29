# distutils: language = c++
# cython: language_level = 3

from libcpp.map cimport map as cmap
from ondepi.resources.ingredients cimport HawkesParam, Sample, EventType
from ondepi.resources.simulation.simulation cimport simulate
from ondepi.resources.likelihood.likelihood  cimport EvalLoglikelihood, eval_loglikelihood

ctypedef cmap[EventType, HawkesParam] QueueParam    

cdef class Queue:
    cdef QueueParam param
    cpdef void set_param(self,
        double alpha_D_0,
        double alpha_D_1, double alpha_D_2,  double beta_D, 
        double alpha_A_0,
        double alpha_A_1, double alpha_A_2, double beta_A,
        double nu_A)
    cpdef Sample simulate(self, double max_time, long unsigned int max_events)
    cpdef void calibrate(self, Sample sample)
