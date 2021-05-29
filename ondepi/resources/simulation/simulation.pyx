# distutils: language = c++

cdef Sample simulate(double max_time, long unsigned int max_events, double alpha_D_0,
                double alpha_D_1, double alpha_D_2, double beta_D, double alpha_A_0,
                double alpha_A_1, double alpha_A_2, double beta_A,
                double nu_A):
  cdef Intensity intensity = Intensity(alpha_D_0, alpha_D_1, alpha_D_2, beta_D, alpha_A_0,
                                  alpha_A_1, alpha_A_2, beta_A, nu_A)
  cdef Sample events_states
  events_states.observations.reserve(max_events)
  cdef EventState first_observation
  first_observation.time=0.0
  first_observation.event = EventType.A
  first_observation.state = 1
  events_states.observations.push_back(first_observation)
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
  while ((events_states.observations.size() < max_events) &
         (intensity.get_time() < max_time)):
    intensity_overall = intensity.sum_()
    uniform_distr.reset()
    U = uniform_distr(generator)
    random_exponential = -log(1.0 - U) / intensity_overall
    time += random_exponential
    ival = intensity.eval_after_last_event(time)
    uniform_distr.reset()
    random_uniform = intensity_overall * uniform_distr(generator)
    intensity.update(ival)
    if (random_uniform < ival[EventType.D] + ival[EventType.A]):
      weights = [ival[EventType.D], ival[EventType.A]]
      distribution.reset()
      distribution = discrete_distribution[int](weights.begin(), weights.end())
      new_.event = EventType.D if distribution(generator) == 0 else EventType.A
      previous_state = events_states.observations.back().state
      new_.state = previous_state - 1 if (new_.event == EventType.D) else previous_state + 1
      new_.time = time
      intensity.arrival(new_)
      events_states.observations.push_back(new_)
  
  return events_states
