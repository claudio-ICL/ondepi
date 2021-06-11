# distutils: language = c++
# cython: language_level = 3

cimport numpy as np
from libc.math cimport log10, ceil, floor
from libcpp.vector cimport vector
from ondepi.resources.ingredients cimport EventType

cdef vector[long] define_event_times(vector[long] time_i, vector[long] dN)
