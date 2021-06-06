from ondepi.resources import utils
from ondepi.resources.likelihood.departures import (
    launch_minimization_D
)
from ondepi.resources.likelihood.arrivals import (
    launch_minimization_A
)
import numpy as np


def generate_random_times_events_states(size=1500):
    times = np.cumsum(np.random.exponential(scale=1.0, size=1500))
    T_end = times[-1] + np.random.exponential(scale=1.0)
    events = np.random.randint(0, 2, size=1500, dtype=int)
    states = np.random.randint(0, 8, size=1500)
    return times, events, states, T_end


def departures():
    print("\n\n*****Departures*****\n")
    print("Test minimization of calibration target D: ")
    times, events, states, T_end = generate_random_times_events_states(
        size=15000)
    res = launch_minimization_D(
        times,
        events,
        states,
        T_end,
        num_guesses=4,
        maxiter=100,
        disp=0)
    print(res)
    best = utils.select_best_optimization_result(res)
    print("\nBest optimization result:")
    print(best)
    return res


def arrivals():
    print("\n\n*****Arrivals*****\n")
    print("Test minimization of calibration target A: ")
    times, events, states, T_end = generate_random_times_events_states(
        size=15000)
    res = launch_minimization_A(
        times,
        events,
        states,
        T_end,
        num_guesses=4,
        maxiter=100,
        disp=0)
    print(res)
    best = utils.select_best_optimization_result(res)
    print("\nBest optimization result:")
    print(best)
    return res


def main():
    res_D = departures()
    res_A = arrivals()
    return 0


if __name__ == '__main__':
    main()
