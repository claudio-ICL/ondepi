# distutils: language = c++
# cython: language_level = 3

from ondepi.resources.ingredients cimport Sample, EventType, EventState
from ondepi.resources.intensity.intensity cimport Intensity, IntensityVal

from libc.math cimport exp, log
from libcpp.vector cimport vector

cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937:
        mt19937()
        mt19937(unsigned int seed)

    cdef cppclass random_device:
        random_device()
        unsigned int operator()()
    
    cdef cppclass default_random_engine:
        default_random_engine()
        default_random_engine(unsigned int seed)

    cdef cppclass uniform_real_distribution[T]:
        uniform_real_distribution()
        uniform_real_distribution(T a, T b)
        T operator()(default_random_engine gen)
        void reset()

    cdef cppclass discrete_distribution[T]:
        discrete_distribution()
        discrete_distribution(vector.iterator first, vector.iterator last)
        T operator()(default_random_engine gen)
        void reset()

cdef Sample simulate(
    double max_time, long unsigned int max_events, 
    double nu_D_0, double nu_D_1, double nu_D_2,
    double alpha_D, double beta_D,
    double nu_A_0, double nu_A_1, double nu_A_2,
    double alpha_A, double beta_A,
    EventType first_event,
    long first_state
)
