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

    def get_variances_of_process(self):
        return self.z_hat.get_variances_of_process()

    cpdef vector[double] get_filter_times(self) except *:
        return self.z_hat.get_times()

    cpdef vector[int] get_filter_dD_t(self) except *:
        return self.z_hat.get_dD_t()

    cpdef vector[int] get_filter_dA_t(self) except *:
        return self.z_hat.get_dA_t()

    cpdef void _set_param(self,
        double nu_D_0 ,double nu_D_1, double nu_D_2,
        double alpha_D, double beta_D,
        double nu_A_0, double nu_A_1, double nu_A_2, 
        double alpha_A, double beta_A
        ) except *:
        self.intensity = Intensity(
            nu_D_0, nu_D_1, nu_D_2,
            alpha_D, beta_D,
            nu_A_0, nu_A_1, nu_A_2, 
            alpha_A, beta_A
        )
        self.param[EventType.D].nu_0 = nu_D_0
        self.param[EventType.D].nu_1 = nu_D_1
        self.param[EventType.D].nu_2 = nu_D_2
        self.param[EventType.D].alpha = alpha_D
        self.param[EventType.D].beta = beta_D
        self.param[EventType.A].nu_0 = nu_A_0
        self.param[EventType.A].nu_1 = nu_A_1
        self.param[EventType.A].nu_2 = nu_A_2
        self.param[EventType.A].alpha = alpha_A
        self.param[EventType.A].beta = beta_A

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
                self.param.at(EventType.D).nu_0,
                self.param.at(EventType.D).nu_1,
                self.param.at(EventType.D).nu_2,
                self.param.at(EventType.D).alpha,
                self.param.at(EventType.D).beta,
                self.param.at(EventType.A).nu_0,
                self.param.at(EventType.A).nu_1,
                self.param.at(EventType.A).nu_2,
                self.param.at(EventType.A).alpha,
                self.param.at(EventType.A).beta,
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
        self.populate_intensity(dt)

        # Populate filter
        self.z_hat = Z_hat()
        self.z_hat.set_sample(this_sample)
        self.z_hat.set_times(self.intensity.get_times())
        self.z_hat.set_dD_t(self.intensity.get_dD_t())
        self.z_hat.set_dA_t(self.intensity.get_dA_t())
        self.z_hat.set_intensity(self.intensity)
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




def produce_df_detection(queue, double beta=1.0, price_level=None,
        idx_precision=9,
        event_times_only=True,
        use_regularisation=False):
    # Expected value from filter 
    df_filter = queue.get_expected_process().reset_index()
    utils.check_nonempty_df(df_filter)
    cdef double dt = df_filter['time'].diff().min()
    cdef long precision = <long>max(1, 10 ** max(idx_precision, (1 - <long>ceil(log10(dt)))))
    df_filter.insert(0, 'idx', np.array(
        np.floor(precision*df_filter['time'].values), dtype=np.int64))

    # Evolution of price queue
    df = queue.get_evolution().reset_index()
    utils.check_nonempty_df(df)
    df.insert(0, 'idx', np.array(
        np.floor(precision*df['time'].values), dtype=np.int64))
    df = df.merge(df_filter, on='idx', how='outer',
                  suffixes=(' sample', ' filter'), validate='1:1')
    df.sort_values(by=['idx'], inplace=True)

    # Variances of the filter
    df_var = queue.get_variances_of_process().reset_index()
    utils.check_nonempty_df(df_var)
    df_var.insert(0, 'idx', np.array(
        np.floor(precision*df_var['time'].values), dtype=np.int64))
    df_var.insert(1, 'std', np.sqrt(df_var['variance'].values))
    df = df.merge(df_var, on='idx', how='outer',
                  suffixes=('', ' variance'), validate='1:1')
    df.sort_values(by=['idx'], inplace=True)

    # Restrict to event times or forward-fill
    if event_times_only:
        df.dropna(inplace=True)
    else:
        df.ffill(inplace=True)

    # Define predictor
    cdef np.ndarray[double, ndim=1] times = np.array(df['time sample'].values, dtype=np.float64)
    cdef np.ndarray[long, ndim=1] states = np.array(df['state'].values, dtype=np.int64)
    cdef np.ndarray[double, ndim=1] z_hat = np.array(df['expected val'].values, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] reg = np.zeros_like(z_hat)
    if use_regularisation:
        reg = regularise_expected_values(times, states, z_hat, beta)
    else:
        reg = z_hat
    df.insert(df.shape[1], 'predictor', reg)

    # Define error
    df.insert(df.shape[1], 'error', df['state'].values - df['predictor'].values)

    # Define detector
    df.insert(df.shape[1], 'detector', df['error'])
    idx = df['variance'] > 0.0
    df.loc[idx, 'detector'] = df.loc[idx, 'error'] / df.loc[idx, 'std']

    # Format result
    df['state'] = df['state'].astype(np.int64)
    cols = ['idx', 'time sample', 'time filter',
            'state', 'expected val', 'variance', 'predictor', 'error', 'detector']
    df = df[cols]
    if price_level is not None:
        cols = [(price_level, col) for col in df.columns]
        df.columns = pd.MultiIndex.from_tuples(cols)
    return df
