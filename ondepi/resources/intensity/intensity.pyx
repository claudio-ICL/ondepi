# distutils: language = c++
# cython: language_level = 3

cdef class Intensity(Process):
    def __cinit__(
            self,
            double alpha_D_0, double alpha_D_1, double alpha_D_2,
            double beta_D, double nu_D,
            double alpha_A_0, double alpha_A_1, double alpha_A_2, 
            double beta_A, double nu_A
        ):
        self.alpha_D = Alpha_D(alpha_D_0, alpha_D_1, alpha_D_2)
        self.beta_D = beta_D
        self.nu_D = nu_D
        self.alpha_A = Alpha_A(alpha_A_0, alpha_A_1, alpha_A_2)
        self.beta_A = beta_A
        self.nu_A = nu_A

    cdef void set_first(self, EventType first_event, long first_state):
        if first_event == EventType.D:
            self.T_D = 0.0
        else:
            self.T_D = -1.0
        cdef EventState first
        first.time=0.0
        first.event = first_event
        first.state = first_state
        self.arrival(first)

    cdef void arrival(self, EventState new_):
        self.update(self.eval_after_last_event(new_.time))
        if new_.event == EventType.D:
            if new_.state == 0:
                self.values[EventType.D] = 0.0
            else:    
                self.values[EventType.D] += self.alpha_D.eval__(new_.state)
            self.values[EventType.A] += self.alpha_A.eval__(new_.state)
            self.set_T_D(new_.time)
        self.set_state(new_.state)    

    cdef IntensityVal eval_after_last_event(self, double t):
        cdef double decay_D = 0.0 if self.T_D < 0.0 else exp(-self.beta_D * (t-self.T_D))
        cdef double decay_A = 0.0 if self.T_D < 0.0 else exp(-self.beta_A * (t-self.T_D))
        cdef IntensityVal res
        if self.state == 0:
            res[EventType.D] = 0.0
        else:    
            res[EventType.D]  = self.nu_D + (self.values[EventType.D] - self.nu_D) * decay_D
        res[EventType.A]  = self.nu_A + (self.values[EventType.A] - self.nu_A) * decay_A
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

