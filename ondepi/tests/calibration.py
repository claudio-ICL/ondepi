from ondepi.resources.queue.queue import Queue


def main():
    queue = Queue()
    queue._set_param(
        alpha_D_0=1.0,
        alpha_D_1=1.0,
        alpha_D_2=1.0,
        beta_D=0.5,
        nu_D=3.0,
        alpha_A_0=1.0,
        alpha_A_1=1.0,
        alpha_A_2=10.0,
        beta_A=0.1,
        nu_A=0.5
    )
    print("\nOriginal param:\n{}".format(queue.get_param()))
    queue.simulate(1000.0, 5000, 1, 2)
    queue_ = Queue()
    queue_.calibrate(
        queue.get_sample(),
        num_guesses=12,
        ftol=1e-14,
        gtol=1e-8,
        maxiter=2000,
        disp=0)
    print("\nEstimated param:\n{}".format(queue_.get_param()))


if __name__ == '__main__':
    main()
