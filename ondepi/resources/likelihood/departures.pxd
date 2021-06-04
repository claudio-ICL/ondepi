# distutils: language = c++
# cython: language_level = 3

from libc.math cimport exp, log
from libcpp.vector cimport vector
cimport numpy as np

from ondepi.resources.ingredients cimport (
    EvalLoglikelihood,
)    

from ondepi.resources.intensity.impact_functions cimport (
        Alpha_D,
        next_v,
        next_v_alpha,
        next_v_beta
)        

cdef EvalLoglikelihood eval_loglikelihood(
         double alpha_D_0, double alpha_D_1, double alpha_D_2, 
         double beta_D, double nu_D,
         vector[double] times,
         vector[long] states,
         double T_end
)

