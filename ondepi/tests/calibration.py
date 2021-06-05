from ondepi.resources.likelihood.departures import (
    calibration_target_D,
    minimize_calibration_target_D,
)    
from ondepi.resources.likelihood.arrivals import (
    calibration_target_A,
    minimize_calibration_target_A,
)    
import numpy as np


def departures():
    print("\n\n*****Departures*****\n")
    params = np.array([
                1.0, 0.5, 1.0,
                1.0, 0.5,
                ], dtype=float)
    times = np.cumsum(np.random.exponential(scale = 1.0, size=1500))
    T_end = times[-1] + np.random.exponential(scale=1.0)
    queue = np.random.randint(0,8, size=1500)
    print("Test evaluation of calibration_target_D: ")
    f, g = calibration_target_D(params, times, queue, T_end)
    print("f = {}".format(f))
    print("g = {}".format(g))

    print("Test minimization of calibration target D: ")
    res = minimize_calibration_target_D(
            times,
            queue,
            T_end,
            maxiter = 40,
            disp=1)
    print(res)
    return res


def arrivals():
    print("\n\n*****Arrivals*****\n")
    params = np.array([
                10.0, 0.5, 2.0,
                1.0, 0.5,
                ], dtype=float)
    times = np.cumsum(np.random.exponential(scale = 1.0, size=1500))
    T_end = times[-1] + np.random.exponential(scale=1.0)
    events = np.random.randint(0,2, size=15000, dtype=int)
    queue = np.random.randint(0,8, size=1500)
    print("Test evaluation of calibration_target_A: ")
    f, g = calibration_target_A(params, times, events, queue, T_end)
    print("f = {}".format(f))
    print("g = {}".format(g))

    print("Test minimization of calibration target A: ")
    res = minimize_calibration_target_A(
            times,
            events,
            queue,
            T_end,
            maxiter = 40,
            disp=1)
    print(res)
    return res

def main():
    res_D = departures()
    res_A = arrivals()
    return 0


if __name__=='__main__':
    main()
