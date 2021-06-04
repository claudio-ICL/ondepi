cdef vector[double] times_D_in_sample(Sample sample):
    cdef long unsigned int num_observations = sample.observations.size()
    cdef vector[double] times
    times.reserve(num_observations)
    cdef long unsigned int i=0
    cdef EventState es
    for i in range(num_observations):
        es = sample.observations.at(i)
        if es.event == EventType.D:
            times.push_back(es.time)
    return times        

cdef vector[long] states_D_in_sample(Sample sample):
    cdef long unsigned int num_observations = sample.observations.size()
    cdef vector[long] states
    states.reserve(num_observations)
    cdef long unsigned int i=0
    cdef EventState es
    for i in range(num_observations):
        es = sample.observations.at(i)
        if es.event == EventType.D:
            states.push_back(es.state)
    return states        
        
        

cdef class Process:
    def __cinit__(self):
        pass

    cdef void set_sample(self, Sample* sample):
        self.sample = sample

    cdef vector[double] get_times(self):
        return self.times

    cdef void set_times(self, vector[double] times):
        self.times = times

    cdef vector[int] get_dD_t(self):
        return self.dD_t

    cdef void set_dD_t(self, vector[int] dD_t):
        self.dD_t = dD_t

    cdef vector[int] get_dA_t(self):
        return self.dA_t

    cdef void set_dA_t(self, vector[int] dA_t):
        self.dA_t = dA_t

    cdef void init_times(self, double dt):
        # Clear vectors
        self.times.clear()
        self.dD_t.clear()
        self.dA_t.clear()

        # Initialise auxiliary variables 
        cdef:
            double t = 0.0
            double T = self.sample[0].observations.front().time
            double dT = self.sample[0].observations.at(1).time - T
            long unsigned int n = 1
            long unsigned int num_observations = self.sample[0].observations.size()
            EventState observation

        # Run
        for n in range(num_observations - 1):
            observation = self.sample[0].observations.at(n)
            T = observation.time
            self.times.push_back(T)
            if (observation.event == EventType.D):
                self.dD_t.push_back(1)
            else:    
                self.dD_t.push_back(0)
            if (observation.event == EventType.A):
                self.dA_t.push_back(1)
            else:    
                self.dA_t.push_back(0)
            next_T = self.sample[0].observations.at(n+1).time
            t = T + dt
            while t <=  next_T - 0.5 * dt:
                self.times.push_back(t)
                self.dD_t.push_back(0)
                self.dA_t.push_back(0)
                t += dt

