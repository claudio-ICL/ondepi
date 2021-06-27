import numpy as np

cdef EvalLoglikelihood eval_loglikelihood(
         double nu_0, double nu_1, double nu_2, 
         double alpha, double beta,
         EventType event_type,
         vector[double] times, # increasing sequence of times T_n when when events D or A happened
         vector[EventType] events, # event occurred at T_n, either D or A
         vector[long] states, # size of the queue at T_n
         double T_end # length of the observation period, assumed to start at 0.0
         ):

    # Impose bounds
    nu_2 = max(0.0, nu_2)
    alpha = max(0.0, alpha)
    beta = max(1.e-6, beta)

    # Instantiate base rate
    cdef BaseRate baserate = BaseRate(nu_0, nu_1, nu_2)
    baserate.set_state(states.front())

    # Define auxiliary variables
    cdef:
        double T
        double next_T
        double dT
        long unsigned int N = times.size()

    # Initialise intensity
    cdef double v_A  = 0.0 # value of sum_{T_D <= t} exp(-beta * ( t - T_D)) evaluated at most recent T_{A}^{k}
    cdef double v_D  = 0.0 # value of sum_{T_D <= t} exp(-beta * ( t - T_D)) evaluated at most recent T_{D}^{j}
    cdef double v_A_beta = 0.0
    cdef double v_D_beta = 0.0
    cdef double lambda_ = baserate.get_value()
    if events.front() == EventType.D:
        v_D = 1.0
        if event_type == EventType.D:
            lambda_ +=  alpha

    # Initialise l_plus
    cdef: 
        double l_plus = 0.0
        double l_plus_nu_0 = 0.0
        double l_plus_nu_1 = 0.0
        double l_plus_nu_2 = 0.0
        double l_plus_alpha = 0.0
        double l_plus_beta = 0.0
    if lambda_ > 0.0:    
        l_plus = log(lambda_)
        l_plus_nu_0 = baserate.get_partial_0() / lambda_
        l_plus_nu_1 = baserate.get_partial_1() / lambda_
        l_plus_nu_2 = baserate.get_partial_2() / lambda_
        if event_type == EventType.A:
            l_plus_alpha = v_A / lambda_
        else:
            l_plus_alpha = v_D / lambda_

  
    # Initialise l_minus
    cdef:
        double decay = 0.0
        double decay_end = 0.0
        double l_minus = 0.0
        double l_minus_base = 0.0
        double l_minus_impact = 0.0
        # The muliplication by alpha/beta will be done at the end
        double l_minus_nu_0 = 0.0
        double l_minus_nu_1 = 0.0
        double l_minus_nu_2 = 0.0
        double l_minus_alpha = 0.0
        double l_minus_beta = 0.0

    T = times.front()
    if events.front() == EventType.D:
        decay_end = compute_decay(T_end, T, beta)
        l_minus_impact = 1.0 - decay_end
        l_minus_alpha = 1.0 - decay_end
        l_minus_beta = (1.0 - decay_end) / beta + (T_end - T) * decay_end


    cdef:
        double new_v_A = v_A
        double new_v_A_beta = v_A_beta
        double new_v_D = v_D
        double new_v_D_beta = v_D_beta
        EventType next_event
        long next_Q
        long unsigned int n

    for n in range(N - 1):    
        T = times[n]
        next_T = times[n + 1]
        next_event = events[n + 1]
        next_Q = states[n + 1]
        v_A = new_v_A
        v_A_beta = new_v_A_beta
        v_D = new_v_D
        v_D_beta = new_v_D_beta

        # Update l_minus_base
        l_minus_base += baserate.get_value() * (next_T - T)
        l_minus_nu_0 += baserate.get_partial_0() * (next_T - T)
        l_minus_nu_1 += baserate.get_partial_1() * (next_T - T)
        l_minus_nu_2 += baserate.get_partial_2() * (next_T - T)

        # Set new state in base rate
        baserate.set_state(next_Q)

        # Update intensity
        if next_event == EventType.D:
            new_v_D = compute_next_v(v_D, T_D, next_T, beta)
            new_v_D_beta = compute_next_v_beta(v_D, v_D_beta, T_D, next_T, beta)
            T_D = next_T
            if event_type == EventType.D:
                lambda_ = baserate.get_value() + alpha * new_v_D
        else:
            T_A = next_T
            new_v_A = v_D * exp(- beta * (T_A - T_D))
            new_v_A_beta = compute_next_v_beta(v_D, v_D_beta, T_D, T_A, beta) # Notice that we pass v_D and v_D_beta to this function 
            if event_type == EventType.A:
                lambda_ = baserate.get_value() + alpha * new_v_A

        # Update l_plus
        if event_type == next_event:
            if lambda_ > 0.0:
                l_plus += log(lambda_)
                l_plus_nu_0 += baserate.get_partial_0() / lambda_
                l_plus_nu_1 += baserate.get_partial_1() / lambda_
                l_plus_nu_2 += baserate.get_partial_2() / lambda_
                if event_type == EventType.A:
                    l_plus_alpha += new_v_A / lambda_
                    l_plus_beta += new_v_A_beta / lambda_
                else:
                    l_plus_alpha += new_v_D / lambda_
                    l_plus_beta += new_v_D_beta / lambda_

        # Update l_minus
        if next_event == EventType.D:
            decay_end = compute_decay(T_end, T_D, beta)
            l_minus_impact += 1.0 - decay_end
            l_minus_alpha += 1.0 - decay_end
            l_minus_beta += (1.0 - decay_end) / beta + (T_end - T_D) + decay_end

    # end of for-loop    

    # multiply l_plus_beta by leading coefficients
    l_plus_beta *= alpha

    # multiply l_minus and its derivatives by leading coefficients
    l_minus_impact *= alpha / beta
    l_minus_alpha *= 1.0 / beta
    l_minus_beta *= alpha / beta

    # Set l_minus
    l_minus = l_minus_base + l_minus_impact

    cdef EvalLoglikelihood res
    res.logL = l_plus - l_minus
    res.gradient = vector[double](5, 0.0)
    res.gradient[0] = l_plus_nu_0 - l_minus_nu_0
    res.gradient[1] = l_plus_nu_1 - l_minus_nu_1
    res.gradient[2] = l_plus_nu_2 - l_minus_nu_2
    res.gradient[3] = l_plus_alpha - l_minus_alpha
    res.gradient[4] = l_plus_beta - l_minus_beta
    return res


def calibration_target(
        np.ndarray[double, ndim=1] params,
        EventType event_type,
        np.ndarray[double, ndim=1] times,
        np.ndarray[long, ndim=1] events,
        np.ndarray[long, ndim=1] states,
        double T_end,
        ):
    cdef:
        double nu_0 = params[0]
        double nu_1 = params[1]
        double nu_2 = max(params[2], 0.0)
        double alpha = max(params[3], 0.0)
        double beta = max(params[4], 0.00000001)
    cdef vector[double] time_vector = times 
    cdef vector[EventType] event_vector = events
    cdef vector[long] state_vector = states 
    cdef EvalLoglikelihood res = eval_loglikelihood(
            nu_0, nu_1, nu_2, 
            alpha, beta,
            event_type,
            time_vector,
            event_vector,
            state_vector,
            T_end
            )
    cdef double f =  - res.logL
    cdef np.ndarray[double, ndim=1] grad = np.array(res.gradient, dtype=float)
    cdef np.ndarray[double, ndim=1] g = - grad
    return f, g

