# distutils: language = c++
# cython: language_level = 3

import numpy as np
import pandas as pd

cdef class Intensity(Process):
    def __cinit__(
            self,
            double nu_D_0, double nu_D_1, double nu_D_2,
            double alpha_D, double beta_D,
            double nu_A_0, double nu_A_1, double nu_A_2, 
            double alpha_A, double beta_A
        ):
        cdef BaseRate baserate_D = BaseRate(nu_D_0, nu_D_1, nu_D_2)
        cdef ImpactRate impactrate_D = ImpactRate(alpha_D, beta_D)
        cdef BaseRate baserate_A = BaseRate(nu_A_0, nu_A_1, nu_A_2)
        cdef ImpactRate impactrate_A = ImpactRate(alpha_A, beta_A)
        self.baserate_D = baserate_D
        self.baserate_A = baserate_A
        self.impactrate_D = impactrate_D
        self.impactrate_A = impactrate_A

    cdef void set_first(self, EventType first_event, long first_state):
        if first_event == EventType.D:
            self.T_D = 0.0
        else:
            self.T_D = -1.0
        cdef EventState first
        first.time = 0.0
        first.event = first_event
        first.state = first_state
        self.arrival(first)

    cdef void arrival(self, EventState new_):
        cdef:
            double impact_D
            double impact_A
            double nu_D
            double nu_A
        if new_.event == EventType.D:
            impact_D = self.impactrate_D.compute_at_arrival(new_.time)
            impact_A = self.impactrate_A.compute_at_arrival(new_.time)
            self.set_T_D(new_.time)
        else:
            impact_D = self.impactrate_D.eval_after_last_event(new_.time)
            impact_A = self.impactrate_A.eval_after_last_event(new_.time)
        nu_D = self.baserate_D.compute_at_arrival(new_.state)
        nu_A = self.baserate_A.compute_at_arrival(new_.state)
        if new_.state == 0:
            self.values[EventType.D] = 0.0
        else:
            self.values[EventType.D] = nu_D + impact_D
        self.values[EventType.A] = nu_A + impact_A    
        self.set_state(new_.state)    

    cdef IntensityVal eval_after_last_event(self, double t):
        cdef double nu_D = self.baserate_D.get_value()
        cdef double nu_A = self.baserate_A.get_value()
        cdef double impact_D = self.impactrate_D.eval_after_last_event(t)
        cdef double impact_A = self.impactrate_A.eval_after_last_event(t)
        cdef IntensityVal res
        if self.state == 0:
            res[EventType.D] = 0.0
        else:    
            res[EventType.D]  = nu_D + impact_D
        res[EventType.A]  = nu_A + impact_A
        return res

    cdef void update(self, IntensityVal values):
        self.values[EventType.D] = values.at(EventType.D)
        self.values[EventType.A] = values.at(EventType.A)

    cdef void set_state(self, long q):
        self.state = q

    cdef long get_state(self):
        return self.state

    cdef void set_T_D(self, double t):
        self.T_D = t

    cdef double get_T_D(self,):
        return self.T_D

    cdef double sum_(self,):
        cdef double lambda_D = self.values.at(EventType.D)
        cdef double lambda_A = self.values.at(EventType.A)
        return lambda_D + lambda_A

    cdef vector[IntensityVal] get_process(self):
        return self.process

    def get_df_process(self, include_baserates=False):
        cdef long unsigned int size = self.process.size()
        cdef np.ndarray[double, ndim=1] times = self.get_times_arr()
        cdef np.ndarray[double, ndim=1] intensity_D = np.zeros(size, dtype=np.float64)
        cdef np.ndarray[double, ndim=1] intensity_A = np.zeros(size, dtype=np.float64)
        cdef np.ndarray[long, ndim=1] states = np.array(self.state_trajectory, dtype=np.int64)
        cdef np.ndarray[double, ndim=1] baserate_D = np.zeros(size, dtype=np.float64)
        cdef np.ndarray[double, ndim=1] baserate_A = np.zeros(size, dtype=np.float64)
        cdef IntensityVal val
        cdef long state
        cdef long unsigned int n
        for n in range(size):
            val = self.process.at(n)
            intensity_D[n] = val.at(EventType.D)
            intensity_A[n] = val.at(EventType.A)
            if include_baserates:
                state = self.state_trajectory.at(n)
                baserate_D[n] = self.baserate_D.eval__(state)
                baserate_A[n] = self.baserate_A.eval__(state)
        if include_baserates:
            df = pd.DataFrame({'time': times, 'state': states, 
                'baserate_D': baserate_D, 'baserate_A': baserate_A,
                'intensity_D': intensity_D, 'intensity_A': intensity_A}
            )
        else:    
            df = pd.DataFrame({'time': times, 'state': states, 
                'intensity_D': intensity_D, 'intensity_A': intensity_A})
        return df

    cdef void init_process(self):
        self.process.clear()
        self.process.reserve(self.times.size())
        # The time of the first observation is assumed as origin
        cdef EventState first_observation = self.sample[0].observations.front()
        self.set_first(first_observation.event, first_observation.state)
        self.process.push_back(self.values)
        cdef vector[long] state_trajectory
        self.state_trajectory.clear()
        self.state_trajectory.reserve(self.times.size())
        self.state_trajectory.push_back(first_observation.state)

    cdef void populate(self):
        # Initialise auxiliary variables
        cdef double next_T
        cdef long unsigned int num_time_points = self.times.size()
        cdef long unsigned int num_observations = self.sample[0].observations.size()
        cdef EventState next_observation
        cdef long unsigned int t = 1
        cdef long unsigned int n = 1 # The first observation is set in init_process

        for t in range(1, num_time_points):
            if (self.dD_t.at(t) == 1) | (self.dA_t.at(t)==1):
                next_observation = self.sample[0].observations.at(n)
                self.state_trajectory.push_back(next_observation.state)
                self.arrival(next_observation)
                self.process.push_back(self.values)
                n += 1
            else:
                self.process.push_back(
                        self.eval_after_last_event(self.times.at(t)))
                self.state_trajectory.push_back(
                        self.get_state())

    cdef double conditional_intensity_from_process(self,
            EventType event_type, 
            long state,
            long unsigned int t
            ):
        if (state < 0) | ((state==0) & (event_type==EventType.D)):
            return 0.0
        cdef double lambda_ = self.process.at(t).at(event_type)
        cdef long historical_state = self.state_trajectory.at(t)
        cdef double pathwise_component = 0.0
        if event_type == EventType.D:
            pathwise_component = lambda_ - self.baserate_D.eval__(historical_state)
        else:
            pathwise_component = lambda_ - self.baserate_A.eval__(historical_state)
        pathwise_component = max(0.0, pathwise_component)
        cdef double baserate_component = 0.0
        if event_type == EventType.D:
            baserate_component = self.baserate_D.eval__(state)
        else:
            baserate_component = self.baserate_A.eval__(state)
        cdef double res = baserate_component + pathwise_component
        return res

    cdef IntensityValForFilterUpdate get_intensities_for_filter_update(self, long state, long unsigned int t):
        cdef IntensityValForFilterUpdate res
        res.lambda_A_below = self.conditional_intensity_from_process(
                EventType.A, state - 1, t)
        res.lambda_A_same = self.conditional_intensity_from_process(
                EventType.A, state, t)
        res.lambda_D_same = self.conditional_intensity_from_process(
                EventType.D, state, t)
        res.lambda_D_above = self.conditional_intensity_from_process(
                EventType.D, state + 1, t)
        return res

    cdef vector[double] get_conditional_lambda_D_intensities_from_process(self, 
            long num_states, long unsigned int t):
        cdef vector[double] res
        res.clear()
        res.reserve(num_states)
        cdef long state
        for state in range(num_states):
            res.push_back(self.conditional_intensity_from_process(
                EventType.D, state, t)
                )
        return res












