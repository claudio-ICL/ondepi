# distutils: language = c++
# cython: language_level = 3

from libc.math cimport exp, log
from libcpp.vector cimport vector
cimport numpy as np

from ondepi.resources.ingredients cimport (
    EventType,
    EvalLoglikelihood,
)    

from ondepi.resources.intensity.impact_functions cimport (
        Alpha_A,
        next_v,
        next_v_alpha,
        next_v_beta
)        

cdef EvalLoglikelihood eval_loglikelihood(
         double alpha_A_0, double alpha_A_1, double alpha_A_2, 
         double beta_A, double nu_A,
         vector[double] times,
         vector[EventType] events, 
         vector[long] states,
         double T_end
)

