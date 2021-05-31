cdef Sample simulate(
    double max_time, long unsigned int max_events, 
    double alpha_D_0, double alpha_D_1, double alpha_D_2,
    double beta_D, double nu_D,
    double alpha_A_0, double alpha_A_1, double alpha_A_2,
    double beta_A, double nu_A,
    EventType first_event,
    long first_state
):
    # Initialise intensity
    cdef Intensity intensity = Intensity(
            alpha_D_0, alpha_D_1, alpha_D_2, 
            beta_D, nu_D, 
            alpha_A_0, alpha_A_1, alpha_A_2, 
            beta_A, nu_A
    )
    intensity.set_first(first_event, first_state)

    # Initialise sample
    cdef Sample sample
    sample.observations.reserve(max_events)
    cdef EventState first_observation
    first_observation.time=0.0
    first_observation.event = first_event
    first_observation.state = first_state
    sample.observations.push_back(first_observation)

    # Initialise auxiliary variables
    cdef double time = 0.0
    cdef IntensityVal ival
    cdef:
        random_device r
        unsigned int seed = r()
        default_random_engine generator  = default_random_engine(seed)
        uniform_real_distribution[double] uniform_distr = uniform_real_distribution[double](0.0, 1.0)
        discrete_distribution[int] distribution
        double intensity_overall
        double U
        double random_exponential
        double random_uniform
        long previous_state
        EventState new_
        vector[double] weights

    # Run    
    while ((sample.observations.size() < max_events) &
           (time < max_time)):
        intensity_overall = intensity.sum_()
        uniform_distr.reset()
        U = uniform_distr(generator)
        random_exponential = -log(1.0 - U) / intensity_overall
        time += random_exponential
        ival = intensity.eval_after_last_event(time)
        uniform_distr.reset()
        random_uniform = intensity_overall * uniform_distr(generator)
        if (random_uniform < ival[EventType.D] + ival[EventType.A]):
            weights.clear()
            weights = [ival[EventType.D], ival[EventType.A]]
            distribution.reset()
            distribution = discrete_distribution[int](weights.begin(), weights.end())
            new_.event = EventType.D if distribution(generator) == 0 else EventType.A
            previous_state = sample.observations.back().state
            new_.state = previous_state - 1 if (new_.event == EventType.D) else previous_state + 1
            new_.time = time
            intensity.arrival(new_)
            sample.observations.push_back(new_)
        else:  
            intensity.update(ival)

    return sample
