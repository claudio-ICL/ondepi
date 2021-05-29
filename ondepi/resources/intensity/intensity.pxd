# distutils: language = c++
# cython: language_level = 3

from libc.math cimport exp, log
from libcpp.map cimport map as cmap
from ondepi.resources.intensity.impact_functions cimport Alpha_D, Alpha_A
from ondepi.resources.ingredients cimport EventType, EventState

ctypedef cmap[EventType, double] IntensityVal    

cdef class Intensity:
    cdef IntensityVal values
    cdef Alpha_D alpha_D
    cdef double beta_D
    cdef Alpha_A alpha_A
    cdef double beta_A
    cdef double nu_A
    cdef double time
    cdef void arrival(self, EventState new_)
    cdef IntensityVal eval_after_last_event(self, double t)
    cdef void update(self, IntensityVal values)
    cdef void set_time(self, double t)
    cdef double get_time(self,)
    cdef double sum_(self,)

