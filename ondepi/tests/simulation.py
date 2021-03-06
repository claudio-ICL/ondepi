from ondepi.resources.queue.queue import Queue


def main():
    queue = Queue()
    queue._set_param(
        nu_D_0=5.0,
        nu_D_1=-1.0,
        nu_D_2=2.0,
        alpha_D=4.0,
        beta_D=3.0,
        nu_A_0=1.0,
        nu_A_1=0.0,
        nu_A_2=3.0,
        alpha_A=1.0,
        beta_A=2.0
    )
    queue.simulate(
        max_time=100.0,
        max_events=10,
        first_event=1,
        first_state=2
    )
    print(queue.get_sample())

    queue.filter(
        dt=0.1,
        num_states=8
    )
    print("\nIntensity process")
    print(queue.get_intensity_process())
    print("\nIntensity times")
    print(queue.get_intensity_times())
    print("\nExpected queue")
    print(queue.get_expected_process())
    print("\nFilter process")
    print(queue.get_filter_process())
    print("\nFilter dD_t")
    print(queue.get_filter_dD_t())
    print("\nFilter dA_t")
    print(queue.get_filter_dA_t())


if __name__ == '__main__':
    main()
