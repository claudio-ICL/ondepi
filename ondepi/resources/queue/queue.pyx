cdef class Queue:
    def __init__(self):
        pass

    def __cinit__(self):
        pass

    cpdef void set_sample(self, Sample sample) except *:
        self.sample = sample

    cpdef Sample get_sample(self) except *:
        return self.sample

    cpdef void set_param(self,
        double alpha_D_0 ,double alpha_D_1, double alpha_D_2,
        double beta_D, double nu_D,
        double alpha_A_0, double alpha_A_1, double alpha_A_2, 
        double beta_A, double nu_A
        ) except *:
        self.intensity = Intensity(
            alpha_D_0, alpha_D_1, alpha_D_2,
            beta_D, nu_D,
            alpha_A_0, alpha_A_1, alpha_A_2, 
            beta_A, nu_A
        )
        self.param[EventType.D].alpha_0 = alpha_D_0
        self.param[EventType.D].alpha_1 = alpha_D_1
        self.param[EventType.D].alpha_2 = alpha_D_2
        self.param[EventType.D].beta = beta_D
        self.param[EventType.D].nu = nu_D
        self.param[EventType.A].alpha_0 = alpha_A_0
        self.param[EventType.A].alpha_1 = alpha_A_1
        self.param[EventType.A].alpha_2 = alpha_A_2
        self.param[EventType.A].beta = beta_A
        self.param[EventType.A].nu = nu_A

    cpdef void simulate(self, 
            double max_time, long unsigned int max_events,
            EventType first_event,
            long first_state
            ) except *:
        self.sample = simulate(
                max_time,
                max_events,
                self.param.at(EventType.D).alpha_0,
                self.param.at(EventType.D).alpha_1,
                self.param.at(EventType.D).alpha_2,
                self.param.at(EventType.D).beta,
                self.param.at(EventType.D).nu,
                self.param.at(EventType.A).alpha_0,
                self.param.at(EventType.A).alpha_1,
                self.param.at(EventType.A).alpha_2,
                self.param.at(EventType.A).beta,
                self.param.at(EventType.A).nu,
                first_event,
                first_state
                )
        self.intensity.set_sample(&(self.sample))

    cpdef void calibrate(self, Sample sample) except *:
        pass
