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

cdef Sample simulate(double max_time, long unsigned int max_events, 
        double alpha_D_0,
        double alpha_D_1, double alpha_D_2,  double beta_D, 
        double alpha_A_0,
        double alpha_A_1, double alpha_A_2, double beta_A,
        double nu_A)
