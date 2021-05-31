from ondepi.resources.queue.queue import Queue


def main():
    queue = Queue()
    queue.set_param(
        alpha_D_0=1.0,
        alpha_D_1=1.0,
        alpha_D_2=1.0,
        beta_D=1.0,
        nu_D=1.0,
        alpha_A_0=1.0,
        alpha_A_1=1.0,
        alpha_A_2=1.0,
        beta_A=1.0,
        nu_A=1.0
    )
    queue.simulate(
        max_time=100.0,
        max_events=5,
        first_event=1,
        first_state=1
    )
    print(queue.get_sample())


if __name__ == '__main__':
    main()
