# distutils: language = c++
# cython: language_level = 3

from libc.math cimport exp, log
from libcpp.vector cimport vector
cimport numpy as np

from ondepi.resources.ingredients cimport (
    EventType,
    EvalLoglikelihood,
)    

from ondepi.resources.intensity.baserate cimport BaseRate
from ondepi.resources.intensity.impactrate cimport(
        compute_decay,
        compute_next_v,
        compute_next_v_beta,
)        

cdef EvalLoglikelihood eval_loglikelihood(
         double nu_0, double nu_1, double nu_2, 
         double alpha, double beta,
         EventType event_type,
         vector[double] times,
         vector[EventType] events, 
         vector[long] states,
         double T_end
)

