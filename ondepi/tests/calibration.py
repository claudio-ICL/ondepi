from ondepi.resources.queue.queue import Queue


def main():
    queue = Queue()
    queue._set_param(
        nu_D_0=5.0,
        nu_D_1=-1.0,
        nu_D_2=1.0,
        alpha_D=3.0,
        beta_D=3.0,
        nu_A_0=10.0,
        nu_A_1=0.0,
        nu_A_2=1.0,
        alpha_A=2.0,
        beta_A=3.5
    )
    print("\nOriginal param:\n{}".format(queue.get_param()))
    queue.simulate(1000.0, 5000, 1, 2)
    queue_ = Queue()
    queue_.calibrate(
        queue.get_sample(),
        num_guesses=12,
        ftol=1e-15,
        gtol=1e-10,
        maxiter=100,
        disp=0)
    print("\nEstimated param:\n{}".format(queue_.get_param()))


if __name__ == '__main__':
    main()
