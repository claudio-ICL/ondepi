import pandas as pd
import numpy as np
from scipy.optimize import minimize
from ondepi.resources.likelihood.calibration import estimate_param
from ondepi.resources import utils

cdef class Queue:
    def __init__(self):
        pass

    def __cinit__(self):
        pass

    cpdef void set_price(self, long price) except *:
        self.price = price

    cpdef long get_price(self) except *:
        return self.price

    cpdef void set_sample(self, Sample sample) except *:
        self.sample = sample

    cpdef Sample get_sample(self) except *:
        return self.sample

    def get_df_sample(self):
        return utils.sample_to_df(self.sample)

    def get_evolution(self):
        df = utils.sample_to_df(self.sample)
        ser = df.loc[:, ['time', 'state']].set_index('time')
        return ser

    cpdef vector[IntensityVal] get_intensity_process(self) except *:
        return self.intensity.get_process()

    cpdef vector[double] get_intensity_times(self) except *:
        return self.intensity.get_times()

    def get_df_intensity_process(self):
        return self.intensity.get_df_process()

    cpdef vector[Z_hat_t] get_filter_process(self) except *:
        return self.z_hat.get_process()

    def get_expected_process(self):
        return self.z_hat.get_expected_process()

    cpdef vector[double] get_filter_times(self) except *:
        return self.z_hat.get_times()

    cpdef vector[int] get_filter_dD_t(self) except *:
        return self.z_hat.get_dD_t()

    cpdef vector[int] get_filter_dA_t(self) except *:
        return self.z_hat.get_dA_t()

    cpdef void _set_param(self,
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

    def set_param(self, 
            np.ndarray[double, ndim=1] d, # array of parameters for the departures 
            np.ndarray[double, ndim=1] a, # array of parameters for the arrivals 
            ):
        self._set_param(
                d[0], d[1], d[2],
                d[3], d[4],
                a[0], a[1], a[2],
                a[3], a[4],
                )

    cpdef QueueParam get_param(self):
        return self.param

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

    cpdef void populate_intensity(self, double dt):
        cdef Sample* this_sample = &(self.sample)
        # Populate history of intensities
        self.intensity.set_sample(this_sample)
        self.intensity.init_times(dt)
        self.intensity.init_process()
        self.intensity.populate()

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

    cpdef void calibrate_on_self(self,
            int num_guesses=5,
            double ftol=1e-12,
            double gtol=1e-6,
            int maxiter=1000, 
            int disp=0,
            launch_async=False, 
            ) except *:
        cdef Sample sample = self.sample
        self.calibrate(
            sample,    
            num_guesses=num_guesses,
            ftol=ftol,
            gtol=gtol,
            maxiter=maxiter, 
            disp=disp,
            launch_async=launch_async)

    cpdef void calibrate(self, 
            Sample sample, 
            int num_guesses=5,
            double ftol=1e-12,
            double gtol=1e-6,
            int maxiter=1000, 
            int disp=0,
            launch_async=False, 
            ) except *:
        cdef double T_end = sample.observations.back().time
        cdef np.ndarray[double, ndim=1] param_D = estimate_param(
                EventType.D, sample, T_end,
                num_guesses=num_guesses,
                ftol=ftol,
                gtol=gtol,
                maxiter=maxiter,
                disp=disp,
                launch_async=launch_async)
        cdef np.ndarray[double, ndim=1] param_A = estimate_param(
                EventType.A, sample, T_end,
                num_guesses=num_guesses,
                ftol=ftol,
                gtol=gtol,
                maxiter=maxiter,
                disp=disp,
                launch_async=launch_async)
        self.set_param(param_D, param_A)




def produce_df_detection(queue, double beta=1.0):
    df_filter = queue.get_expected_process().reset_index()
    utils.check_nonempty_df(df_filter)
    cdef double dt = df_filter['time'].diff().min()
    cdef long precision = <long>max(1, 10 ** (1 - <long>ceil(log10(dt))))
    df_filter.insert(0, 'idx', np.array(
        np.floor(precision*df_filter['time'].values), dtype=np.int64))
    df = queue.get_evolution().reset_index()
    utils.check_nonempty_df(df)
    df.insert(0, 'idx', np.array(
        np.floor(precision*df['time'].values), dtype=np.int64))
    df = df.merge(df_filter, on='idx', how='outer',
                  suffixes=(' sample', ' filter'), validate='1:1')
    df.dropna(inplace=True)
    cdef np.ndarray[double, ndim=1] times = np.array(df['time sample'].values, dtype=np.float64)
    cdef np.ndarray[long, ndim=1] states = np.array(df['state'].values, dtype=np.int64)
    cdef np.ndarray[double, ndim=1] z_hat = np.array(df['expected val'].values, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] reg = regularise_expected_values(times, states, z_hat, beta)
    df.insert(df.shape[1], 'predictor', reg)
    df.insert(df.shape[1], 'error', df['state'].values - df['predictor'].values)
    df['state'] = df['state'].astype(np.int64)
    cols = ['idx', 'time sample', 'time filter',
            'state', 'expected val', 'predictor', 'error']
    df = df[cols]
    return df
