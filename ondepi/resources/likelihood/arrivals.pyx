import numpy as np
from ondepi.resources.likelihood.loglikelihood import calibration_target
from ondepi.resources.likelihood.minimization import minimize_calibration_target, launch_minimization


def calibration_target_A(
        np.ndarray[double, ndim=1] params,
        np.ndarray[double, ndim=1] times,
        np.ndarray[long, ndim=1] events,
        np.ndarray[long, ndim=1] states,
        double T_end,
        ):
    return calibration_target(
            params,
            EventType.A,
            times,
            events,
            states,
            T_end)


def minimize_calibration_target_A(
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
    return minimize_calibration_target(
        EventType.A,
        times,
        events,
        states,
        T_end,
        init_guess, 
        ftol=ftol,
        gtol=gtol,
        maxiter=maxiter,
        disp=disp,
        )

def launch_minimization_A(
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
    return launch_minimization(
        EventType.A,
        times,
        events,
        states,
        T_end,
        num_guesses=num_guesses,
        ftol=ftol,
        gtol=gtol,
        maxiter=maxiter,
        disp=disp,
        launch_async=launch_async,
        )
