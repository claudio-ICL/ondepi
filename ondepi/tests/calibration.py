from ondepi.resources.likelihood.departures import calibration_target_D, minimize_calibration_target_D
import numpy as np


def main():
    print("\n\n*****Departures*****\n")
    params = np.array([
                1.0, 0.5, 1.0,
                1.0, 0.5,
                ], dtype=float)
    times = np.cumsum(np.random.exponential(scale = 1.0, size=5000))
    T_end = times[-1] + np.random.exponential(scale=1.0)
    queue = np.random.randint(0,8, size=5000)
    print("Test evaluation of calibration_target_D: ")
    f, g = calibration_target_D(params, times, queue, T_end)
    print("f = {}".format(f))
    print("g = {}".format(g))

    print("Test minimization of calibration target D: ")
    res = minimize_calibration_target_D(
            times,
            queue,
            T_end,
            maxiter = 100,
            disp=1)
    print(res)
    return res



if __name__=='__main__':
    main()
