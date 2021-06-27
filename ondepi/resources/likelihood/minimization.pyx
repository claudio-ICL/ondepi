import numpy as np
from scipy.optimize import minimize
from ondepi.resources.likelihood.loglikelihood import calibration_target
from ondepi.resources import utils


def minimize_calibration_target(
        EventType event_type, 
        np.ndarray[double, ndim=1] times,
        np.ndarray[long, ndim = 1] events,
        np.ndarray[long, ndim=1] states,
        double T_end,
        np.ndarray[double, ndim=1] init_guess, 
        double ftol=1e-12,
        double gtol=1e-6,
        int maxiter=1000, 
        int disp=0
        ):        
    cdef list bounds = [
            (None, None), # Bounds for nu_0
            (None, None), # Bounds for nu_1
            (0.0, None), # Bounds for nu_2
            (0.0, None), # Bounds for alpha
            (0.0000001, None), # Bounds for beta
    ]
    res = minimize(
            calibration_target,
            init_guess,
            args = (event_type, times, events, states, T_end),
            method = 'L-BFGS-B',
            bounds = bounds, 
            jac = True,
            options = {
                'ftol': ftol,
                'gtol': gtol,
                'maxiter': maxiter,
                'disp': disp}
            )
    return res


def generate_init_guesses(
        np.ndarray[double, ndim=1] times,
        np.ndarray[long, ndim=1] events,
        long event=1,
        int num=5
        ):
    cdef long unsigned int n = 0
    cdef np.ndarray[double, ndim=1] T_event = utils.extract_times_of_event(times, events, event=event)
    cdef double interarrival_avg = np.mean(np.diff(T_event))
    cdef list nu_0s = np.random.uniform(-10.0, 10.0, size=num).tolist()
    cdef list nu_1s = np.random.uniform(-10.0, 10.0, size=num).tolist()
    cdef list alphas = np.random.uniform(0.0, 10.0, size=num).tolist()
    cdef list betas = np.random.uniform(0.1, 2.0, size=num).tolist()
    cdef list init_guesses = []
    cdef np.ndarray[double, ndim=1] param = np.ones(5, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] guess = np.ones(5, dtype=np.float64)
    for nu_0, nu_1, alpha, beta in zip(nu_0s, nu_1s, alphas, betas):
        param[0] = nu_0
        param[1] = nu_1
        param[2] = 1.0 / interarrival_avg
        param[3] = alpha
        param[4] = beta
        guess = np.array(param, copy=True)
        init_guesses.append(guess)
        param[2] = 0.01 / interarrival_avg
        guess = np.array(param, copy=True)
        init_guesses.append(guess)
    return init_guesses   



def launch_minimization(
        EventType event_type, 
        np.ndarray[double, ndim=1] times,
        np.ndarray[long, ndim=1] events,
        np.ndarray[long, ndim=1] states,
        double T_end,
        int num_guesses=5,
        double ftol=1e-12,
        double gtol=1e-6,
        int maxiter=1000, 
        int disp=0,
        launch_async=False
        ):
    cdef list init_guesses = generate_init_guesses(
        times,
        events,
        event=1,
        num=num_guesses,
        )
    cdef list list_args = [
            (event_type, times, events, states, T_end, init_guess, ftol, gtol, maxiter, disp)
            for init_guess in init_guesses
            ]
    cdef list res
    if launch_async:
        res = utils.launch_async(minimize_calibration_target, list_args)
    else:    
        res = utils.launch_serial(minimize_calibration_target, list_args)
    return res
