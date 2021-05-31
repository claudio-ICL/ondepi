cdef class Process:
    def __cinit__(self):
        pass

    cdef void set_sample(self, Sample* sample):
        self.sample = sample

    cdef void init_times(self, double dt):
        self.times.clear()
        self.dD_t.clear()
        cdef:
            double t = 0.0
            double T = self.sample[0].observations.front().time
            double dT = self.sample[0].observationsu.at(1).time - T
            long unsigned int n = 1
        for n in range(1, self.sample[0].observations.size()):
            next_T = self.sample[0].observations.at(n).time
            dT = min(dT, next_T - T)
            T = next_T
        dt = min(dt, dT)    
        for n in range(self.sample[0].observations.size() - 1):
            T = self.sample[0].observations.at(n).time
            next_T = self.sample[0].observations.at(n+1).time
            t = T
            while t <=  next_T - 0.5 * dt:
                self.times.push_back(t)
                if t < T + 0.5 * dt:
                    self.dD_t.push_back(1)
                else:    
                    self.dD_t.push_back(0)
                t += dt

