# distutils: language = c++
# cython: language_level = 3

from libc.math cimport exp, log
from libcpp.vector cimport vector
from libcpp.map cimport map as cmap
from ondepi.resources.intensity.impact_functions cimport Alpha_D, Alpha_A
from ondepi.resources.ingredients cimport EventType, EventState, Process

ctypedef cmap[EventType, double] IntensityVal    

cdef class Intensity(Process):
    cdef IntensityVal values
    cdef vector[IntensityVal] process
    cdef void init_process(self)
    cdef void populate(self)
    cdef vector[IntensityVal] get_process(self)
    cdef Alpha_D alpha_D
    cdef double beta_D
    cdef double nu_D
    cdef Alpha_A alpha_A
    cdef double beta_A
    cdef double nu_A
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

