import numpy as np
import pandas as pd


cdef double update_0(double z_hat_t_0, double z_hat_t_1, 
    IntensityValForFilterUpdate intensity,
    double mu,
    int dD_t,
    double dt
    ):
    cdef double predictable = (
            - z_hat_t_0 * intensity.lambda_A_same * dt +
            z_hat_t_1 * intensity.lambda_D_above * dt
            )
    cdef double innovation = (
            (intensity.lambda_D_above * z_hat_t_1 - z_hat_t_0 * mu) * 
            (dD_t / mu - dt)
            )
    cdef double  res = z_hat_t_0 + predictable + innovation
    return max(0.0, res)    


cdef double update_local(
    Z_hat_t_local z_hat_t,
    IntensityValForFilterUpdate intensity,
    double mu,
    int dD_t,
    double dt
    ):
    cdef double predictable = (
            z_hat_t[Neighbours._below] * intensity.lambda_A_below * dt -
            z_hat_t[Neighbours._same] * (intensity.lambda_A_same + intensity.lambda_D_same) * dt +
            z_hat_t[Neighbours._above] * intensity.lambda_D_above * dt
            )
    cdef double innovation = 0.0
    if mu > 0.0:        
        innovation = (
            (intensity.lambda_D_above * z_hat_t[Neighbours._above] - z_hat_t[Neighbours._same] * mu) * 
            (dD_t / mu - dt)
        )
    else:
        innovation = (
            (intensity.lambda_D_above * z_hat_t[Neighbours._above] ) * 
            (- dt)
        )
    cdef double  res = z_hat_t[Neighbours._same] + predictable + innovation
    return max(0.0, res)    


cdef class Z_hat(Process):

    cdef void init_process(self):
        self.process.clear()
        self.process.reserve(self.times.size())

    cdef vector[Z_hat_t] get_process(self):
        return self.process

    def get_expected_process(self):
        cdef long unsigned int size = self.process.size()
        cdef vector[double] time
        time.reserve(size)
        cdef vector[double] val
        val.reserve(size)
        cdef long unsigned int t
        for t in range(size):
            time.push_back(self.process[t].time)
            val.push_back(self.process[t].expected_value)
        df = pd.DataFrame({
            'time': time,
            'expected val': val})
        ser = df.set_index('time')
        return ser

    cdef void set_intensity(self, Intensity intensity):
        self.intensity = intensity
    
    cdef Intensity get_intensity(self):
        return self.intensity

    cdef Z_hat_t get_slice(self, long unsigned int idx):
        return self.process.at(idx)

    cdef Z_hat_t get_time_slice(self, double t):
        cdef long unsigned int n = 0
        cdef long unsigned int N = self.process.size()
        if N > 2:
            for n in range(N - 1):
                if (self.process.at(n+1).time > t) & (self.process.at(n).time <= t):
                    return self.get_slice(n)
        cdef Z_hat_t res
        return res

    cdef void populate(self, long unsigned int num_states):
        self.init_process()

        # Initial value of Z_hat
        cdef long unsigned int initial_state = <long unsigned int>self.sample[0].observations.front().state
        num_states = max(num_states, 1 + initial_state)
        cdef Z_hat_t z_hat_0
        z_hat_0.time = self.sample[0].observations.front().time
        z_hat_0.expected_value = <double>initial_state
        z_hat_0.distribution = vector[double](num_states, 0.0)
        cdef long unsigned int n
        for n in range(num_states):
            if n == initial_state:
                z_hat_0.distribution[n] = 1.0
        self.process.push_back(z_hat_0)        

        # Initialise auxiliary variables 
        cdef Z_hat_t new_z_hat_t
        cdef Z_hat_t_local z_hat_t_local
        cdef double z_hat_t_0
        cdef Z_hat_t previous_filter
        cdef vector[double] new_distribution = vector[double](num_states, 0.0)
        cdef vector[double] conditional_lambda_D = vector[double](num_states, 0.0)
        cdef double mass
        cdef double new_expected_value 
        cdef double mu = 0.0
        cdef long unsigned int t

        # Run
        for t in range(-1 + self.times.size()):
            dt = self.times.at(t+1) - self.times.at(t)
            previous_filter = self.process.back()

            # Compute mu
            mu = 0.0
            conditional_lambda_D.clear()
            conditional_lambda_D = self.intensity.get_conditional_lambda_D_intensities_from_process(
                    num_states, t)
            for n in range(1, num_states):
                mu += previous_filter.distribution[n] * conditional_lambda_D[n]

            # Compute new distribution 
            new_distribution = vector[double](num_states, 0.0)
            new_expected_value = 0.0
            new_distribution[0] = update_0(
                previous_filter.distribution.at(0),
                previous_filter.distribution.at(1),
                self.intensity.get_intensities_for_filter_update(0, t),
                mu,
                self.dD_t.at(t + 1),
                dt)
            mass = new_distribution.front()
            for n in range(1, num_states):
                z_hat_t_local[Neighbours._below] = previous_filter.distribution.at(n - 1)
                z_hat_t_local[Neighbours._same] = previous_filter.distribution.at(n)
                if n >= num_states - 1:
                    z_hat_t_local[Neighbours._above] = 0.0
                else:    
                    z_hat_t_local[Neighbours._above] = previous_filter.distribution.at(n + 1)
                new_distribution[n] = update_local(
                        z_hat_t_local,
                        self.intensity.get_intensities_for_filter_update(n, t),
                        mu, 
                        self.dD_t.at(t + 1),
                        dt)
                mass += new_distribution[n]
                new_expected_value += n * new_distribution[n]
                # End n-indexed for-loop (computation of new distribution)

            # Normalise mass    
            if mass > 0.0:    
                new_expected_value /=  mass 
                for n in range(num_states):
                    new_distribution[n] /= mass

            # Push back to process
            new_z_hat_t.distribution = new_distribution        
            new_z_hat_t.expected_value = new_expected_value
            new_z_hat_t.time = self.times.at(t + 1)
            self.process.push_back(new_z_hat_t)

        # End t-indexed for-loop    


cpdef np.ndarray[double, ndim=1] regularise_expected_values(
        np.ndarray[double, ndim=1] times, 
        np.ndarray[long, ndim=1] states, 
        np.ndarray[double, ndim=1] z_hat, 
        double beta
        ):
    cdef vector[double] times_vector = times.tolist()
    cdef vector[long] states_vector = states.tolist()
    cdef vector[double] z_hat_vector = z_hat.tolist()
    cdef np.ndarray[double, ndim=1] res = np.array(
            _regularise_expected_values(times_vector, states_vector,
                z_hat_vector, beta),
            dtype=np.float64
        )
    return res
    


cdef vector[double] _regularise_expected_values(
        vector[double] times, 
        vector[long] states,
        vector[double] expected_values,
        double beta
        ):
    cdef long unsigned int size = states.size()
    assert size == times.size()
    assert size == expected_values.size()
    cdef vector[double] res
    res.reserve(size)
    cdef vector[double] F
    F.reserve(size)
    cdef vector[double] M
    M.reserve(size)
    cdef double decay
    cdef double error
    cdef long unsigned int t
    error = (states.front() - expected_values.front())**2
    if error <= 0.0:
        F.push_back(expected_values.front())
        M.push_back(1.0)
        res.push_back(expected_values.front())
    else:
        F.push_back(expected_values.front() / error)
        M.push_back(1.0 / error)
        res.push_back(expected_values.front())
    for t in range(size - 1):
        decay = exp(-beta * (times.at(t + 1) - times.at(t)))
        error = (states.at(t + 1) - expected_values.at(t + 1))**2
        if error <= 0.0:
            F.push_back(expected_values.at(t + 1))
            M.push_back(1.0)
            res.push_back(F.at(t + 1))
        else:
            F.push_back(F.at(t) * decay + expected_values.at(t + 1) / error)
            M.push_back(M.at(t) * decay + 1.0 / error)
            res.push_back(F.at(t + 1) / M.at(t + 1))
    return res







