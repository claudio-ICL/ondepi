# distutils: language = c++
# cython: language_level = 3

cdef class Intensity:
    def __cinit__(
            self,
            double alpha_D_0, double alpha_D_1, double alpha_D_2, double beta_D, 
            double alpha_A_0,
            double alpha_A_1, double alpha_A_2, double beta_A, double nu_A
        ):
        self.alpha_D = Alpha_D(alpha_D_0, alpha_D_1, alpha_D_2)
        self.beta_D = beta_D
        self.alpha_A = Alpha_A(alpha_A_0, alpha_A_1, alpha_A_2)
        self.beta_A = beta_A
        self.nu_A = nu_A
        self.time = 0.0
        self.values[EventType.D] = 0
        self.values[EventType.A] =  nu_A
    cdef void arrival(self, EventState new_):
        if new_.event == EventType.D:
            self.values[EventType.D] = self.alpha_D.eval__(new_.state)
            self.values[EventType.A] = self.alpha_A.eval__(new_.state)
        self.set_time(new_.time)
    cdef IntensityVal eval_after_last_event(self, double t):
        cdef IntensityVal res
        res[EventType.D]  = self.values[EventType.D] * exp(-self.beta_D * (t-self.time))
        res[EventType.A]  = self.nu_A + (self.values[EventType.A] - self.nu_A) * exp(-self.beta_A * (t-self.time))
        return res
    cdef void update(self, IntensityVal values):
        self.values[EventType.D] = values.at(EventType.D)
        self.values[EventType.A] = values.at(EventType.A)
    cdef void set_time(self, double t):
        self.time = t
    cdef double get_time(self,):
        cdef double time = self.time
        return time
    cdef double sum_(self,):
        cdef double lambda_D = self.values.at(EventType.D)
        cdef double lambda_A = self.values.at(EventType.A)
        return lambda_D + lambda_A

