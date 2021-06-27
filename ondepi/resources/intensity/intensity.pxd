# distutils: language = c++
# cython: language_level = 3

from libc.math cimport exp, log
from libcpp.vector cimport vector
from libcpp.map cimport map as cmap
cimport numpy as np
from ondepi.resources.intensity.baserate cimport BaseRate
from ondepi.resources.intensity.impactrate cimport ImpactRate
# from ondepi.resources.intensity.impactrate  cimport compute_next_v, compute_next_v_beta, compute_decay
from ondepi.resources.ingredients cimport EventType, EventState, Process

ctypedef cmap[EventType, double] IntensityVal    

cdef struct IntensityValForFilterUpdate:
    double lambda_A_below
    double lambda_A_same
    double lambda_D_same
    double lambda_D_above

# ctypedef cmap[EventType, BaseRate] BaseRates
# ctypedef cmap[EventType, ImpactRate] ImpactRates
# cdef struct BaseRates:
#     BaseRate D
#     BaseRate A
# cdef struct ImpactRates:
#     ImpactRate D
#     ImpactRate A

cdef class Intensity(Process):
    cdef BaseRate baserate_D
    cdef BaseRate baserate_A
    cdef ImpactRate impactrate_D
    cdef ImpactRate impactrate_A
    cdef IntensityVal values
    cdef vector[IntensityVal] process
    cdef vector[long] state_trajectory
    cdef void init_process(self)
    cdef void populate(self)
    cdef vector[IntensityVal] get_process(self)
    cdef long state
    cdef double T_D
    cdef void set_first(self, EventType first_event, long first_state)
    cdef void arrival(self, EventState new_)
    cdef IntensityVal eval_after_last_event(self, double t)
    cdef void update(self, IntensityVal values)
    cdef void set_state(self, long q)
    cdef long get_state(self,)
    cdef void set_T_D(self, double t)
    cdef double get_T_D(self,)
    cdef double sum_(self,)
    cdef double conditional_intensity_from_process(self,
            EventType event_type, 
            long state,
            long unsigned int t)
    cdef IntensityValForFilterUpdate get_intensities_for_filter_update(
            self, long state, long unsigned int t)
    cdef vector[double] get_conditional_lambda_D_intensities_from_process(self, 
            long num_states, long unsigned int t)

