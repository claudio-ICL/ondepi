from ondepi.resources.likelihood.departures import launch_minimization_D
from ondepi.resources.likelihood.arrivals import launch_minimization_A
from ondepi.resources import utils

def estimate_param(
        EventType direction,
        Sample sample, 
        double T_end,
        int num_guesses=5,
        double ftol=1e-12,
        double gtol=1e-6,
        int maxiter=1000, 
        int disp=0,
        launch_async=False, 
        ):
    times, events, states = utils.sample_to_arrays(sample)
    fun = launch_minimization_D if direction == EventType.D else launch_minimization_A
    results = fun(
            times, events, states,
            T_end, 
            num_guesses=num_guesses,
            ftol=ftol,
            gtol=gtol,
            maxiter=maxiter,
            disp=disp,
            launch_async=launch_async)
    best = utils.select_best_optimization_result(results)
    cdef str info = "\nEstimated parameters for event type {}:\n{}".format(direction, best)
    utils.logger.info(info)
    return best.get('x')
