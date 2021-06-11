import pandas as pd

cdef double update_0(double z_hat_t_0, double z_hat_t_1, IntensityVal ival, int dD_t, double dt):
    cdef double mu_ = ival.at(EventType.D)
    cdef double lambda_ = ival.at(EventType.A)
    cdef predictable = (
            - z_hat_t_0 * lambda_ * dt +
            z_hat_t_0 * (1.0 -  z_hat_t_0) * mu_ * dt
            )
    cdef innovation = 0.0
    if z_hat_t_0 == 1.0:
        return z_hat_t_0 + predictable
    else:
        innovation = (
           - z_hat_t_0 * dD_t + 
           z_hat_t_1 * dD_t / (1.0 - z_hat_t_0)
           )
        return z_hat_t_0 + predictable + innovation


cdef double update_local(
        Z_hat_t_local z_hat_t, 
        double z_hat_t_0,
        IntensityVal ival,
        int dD_t,
        double dt
        ):
    cdef double mu_ = ival.at(EventType.D)
    cdef double lambda_ = ival.at(EventType.A)
    cdef predictable = (
            z_hat_t[Neighbours._below] * lambda_ * dt -
            z_hat_t[Neighbours._same] * lambda_ * dt - 
            z_hat_t[Neighbours._same] * z_hat_t_0 * mu_ * dt
            )
    cdef innovation = 0.0
    if z_hat_t_0 == 1.0:
        return z_hat_t[Neighbours._same] + predictable
    else:
        innovation = (
            - z_hat_t[Neighbours._same] * dD_t + 
            z_hat_t[Neighbours._above] * dD_t / (1.0 - z_hat_t_0)
            )
        return z_hat_t[Neighbours._same] + predictable + innovation


cdef class Z_hat(Process):
#    def __cinit__(self):
#        pass

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

    cdef void set_intensities(self, vector[IntensityVal] intensities):
        # It is assumed that the n-th entry of the vector 'intesities'
        # (i.e. self.intensities[n])
        # corresponds to the value of the intensity at time self.times[n]
        self.intensities = intensities
    
    cdef vector[IntensityVal] get_intensities(self):
        return self.intensities

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
        cdef double new_expected_value 

        # Run
        for t in range(-1 + self.times.size()):
            dt = self.times.at(t+1) - self.times.at(t)
            previous_filter = self.process.back()
            new_distribution = vector[double](num_states, 0.0)
            new_expected_value = 0.0
            new_distribution[0] = update_0(
                previous_filter.distribution.at(0),
                previous_filter.distribution.at(1),
                self.intensities.at(t),
                self.dD_t.at(t + 1),
                dt)
            z_hat_t_0 = previous_filter.distribution.at(0)
            for n in range(1, num_states):
                z_hat_t_local[Neighbours._below] = previous_filter.distribution.at(n - 1)
                z_hat_t_local[Neighbours._same] = previous_filter.distribution.at(n)
                if n >= num_states - 1:
                    z_hat_t_local[Neighbours._above] = 0.0
                else:    
                    z_hat_t_local[Neighbours._above] = previous_filter.distribution.at(n + 1)
                new_distribution[n] = update_local(
                        z_hat_t_local,
                        z_hat_t_0,
                        self.intensities.at(t),
                        self.dD_t.at(t + 1),
                        dt)
                new_expected_value += n * new_distribution[n]
            new_z_hat_t.distribution = new_distribution        
            new_z_hat_t.expected_value = new_expected_value
            new_z_hat_t.time = self.times.at(t + 1)
            self.process.push_back(new_z_hat_t)
