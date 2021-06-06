from scipy.optimize import minimize

cdef class Queue:
    def __init__(self):
        pass

    def __cinit__(self):
        pass

    cpdef void set_price(self, long price) except *:
        self.price = price

    cpdef void set_sample(self, Sample sample) except *:
        self.sample = sample

    cpdef Sample get_sample(self) except *:
        return self.sample

    cpdef vector[IntensityVal] get_intensity_process(self) except *:
        return self.intensity.get_process()

    cpdef vector[double] get_intensity_times(self) except *:
        return self.intensity.get_times()

    cpdef vector[Z_hat_t] get_filter_process(self) except *:
        return self.z_hat.get_process()

    cpdef vector[double] get_filter_times(self) except *:
        return self.z_hat.get_times()

    cpdef vector[int] get_filter_dD_t(self) except *:
        return self.z_hat.get_dD_t()

    cpdef vector[int] get_filter_dA_t(self) except *:
        return self.z_hat.get_dA_t()

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

    cdef void _filter(self, double dt, long unsigned int num_states):
        cdef Sample* this_sample = &(self.sample)
        # Populate history of intensities
        self.intensity.set_sample(this_sample)
        self.intensity.init_times(dt)
        self.intensity.init_process()
        self.intensity.populate()

        # Populate filter
        self.z_hat = Z_hat()
        self.z_hat.set_sample(this_sample)
        self.z_hat.set_times(self.intensity.get_times())
        self.z_hat.set_dD_t(self.intensity.get_dD_t())
        self.z_hat.set_dA_t(self.intensity.get_dA_t())
        self.z_hat.set_intensities(self.intensity.get_process())
        self.z_hat.init_process()
        self.z_hat.populate(num_states)

    cpdef void filter(self, double dt, long unsigned int num_states) except *:
        self._filter(dt, num_states)

    cpdef void calibrate(self, 
            Sample sample, 
            int maxiter = 100000, 
            float xtol = 1e-5, 
            int disp=0
            ) except *:
        pass
