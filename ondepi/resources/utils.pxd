# distutils: language = c++
# cython: language_level = 3

from libcpp.vector cimport vector
cimport numpy as np
from ondepi.resources.ingredients cimport EventType, EventState, Sample

cpdef Sample arrays_to_sample(vector[double] times, vector[EventType] events, vector[long] states)

