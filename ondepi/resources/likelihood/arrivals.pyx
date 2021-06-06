from numpy import array as nparray
from scipy.optimize import minimize
from ondepi.resources import utils

cdef EvalLoglikelihood eval_loglikelihood(
         double alpha_A_0, double alpha_A_1, double alpha_A_2, 
         double beta_A, double nu_A,
         vector[double] times, # increasing sequence of times T_n when when events D or A happened
         vector[EventType] events, # event occurred at T_n, either D or A
         vector[long] states, # size of the queue at T_n
         double T_end # length of the observation period, assumed to start at 0.0
         ):
    # Instantiate impact functions
    cdef Alpha_A alpha_A = Alpha_A(alpha_A_0, alpha_A_1, alpha_A_2)

    # Define auxiliary variables
    cdef long unsigned int N = times.size()

    cdef double v_A
    cdef double v_A_alpha_0
    cdef double v_A_alpha_1
    cdef double v_A_alpha_2
    cdef double v_A_beta
    cdef double lambda_A

    # Set first entries
    if events.front() == EventType.D:
        v_A = alpha_A.eval__(states.front())
        v_A_alpha_0 = alpha_A.partial_0_(states.front())
        v_A_alpha_1 = alpha_A.partial_1_(states.front())
        v_A_alpha_2 = alpha_A.partial_2_(states.front())
        v_A_beta = 0.0
    else:    
        v_A = 0.0
        v_A_alpha_0 = 0.0
        v_A_alpha_1 = 0.0
        v_A_alpha_2 = 0.0
        v_A_beta = 0.0

    lambda_A = v_A + nu_A
  
    cdef:
        double l_plus = log(lambda_A)
        double decay = 0.0
        double decay_end = 0.0
        double l_minus = 0.0
        # The muliplication by 1/beta_A will be done at the end
      
        double l_plus_alpha_0 = 0.0
        double l_plus_alpha_1 = 0.0
        double l_plus_alpha_2 = 0.0
        double l_plus_beta = 0.0
        double l_plus_nu = 0.0

        double l_minus_alpha_0 = 0.0
        double l_minus_alpha_1 = 0.0
        double l_minus_alpha_2 = 0.0
        double l_minus_beta = 0.0
        double l_minus_nu = T_end

    if events.front() == EventType.D:
        decay_end = exp(-beta_A * (T_end - times.front()))
        l_minus = alpha_A.eval__(states.front()) * (1.0 - decay_end)
        # The muliplication by 1/beta_A will be done at the end
        l_minus_alpha_0 = alpha_A.partial_0_(states.front()) * (1.0 - decay_end)
        l_minus_alpha_1 = alpha_A.partial_1_(states.front()) * (1.0 - decay_end)
        l_minus_alpha_2 = alpha_A.partial_2_(states.front()) * (1.0 - decay_end)
        l_minus_beta = - l_minus / beta_A + l_minus * (T_end - times.front())

    else:    
        l_plus_nu = 1.0 / lambda_A

    cdef:
        double new_v_A = v_A
        double new_v_A_alpha_0 = v_A_alpha_0
        double new_v_A_alpha_1 = v_A_alpha_1
        double new_v_A_alpha_2 = v_A_alpha_2
        double new_v_A_beta = v_A_beta
        double new_lambda_A = lambda_A
        double T
        double next_T
        EventType next_event
        long Q
        long next_Q
        double next_alpha_A
        double update_l_minus
        long unsigned int n

    for n in range(N - 1):    
        T = times[n]
        next_T = times[n + 1]
        next_event = events[n + 1]
        Q = states[n]
        next_Q = states[n + 1]
        next_alpha_A = alpha_A.eval__(next_Q)
        v_A = new_v_A
        v_A_alpha_0 = new_v_A_alpha_0
        v_A_alpha_1 = new_v_A_alpha_1
        v_A_alpha_2 = new_v_A_alpha_2
        v_A_beta = new_v_A_beta
        lambda_A = new_lambda_A
        if next_event == EventType.D:
            new_v_A = next_v(v_A, T, next_T, next_alpha_A, beta_A)
            new_v_A_beta = next_v_beta(v_A_beta, v_A, T, next_T, beta_A)
            new_v_A_alpha_0 = next_v_alpha(
                v_A_alpha_0, T, next_T,
                alpha_A.partial_0_(next_Q), beta_A
                )
            new_v_A_alpha_1 = next_v_alpha(
                v_A_alpha_1, T, next_T,
                alpha_A.partial_1_(next_Q), beta_A
                )
            new_v_A_alpha_2 = next_v_alpha(
                v_A_alpha_2, T, next_T,
                alpha_A.partial_2_(next_Q), beta_A
                )
        else:    
            decay = exp(-beta_A * (next_T - T))
            new_v_A = v_A * decay
            new_v_A_beta = next_v_beta(v_A_beta, v_A, T, next_T, beta_A)
            new_v_A_alpha_0 = v_A_alpha_0 * decay
            new_v_A_alpha_1 = v_A_alpha_1 * decay
            new_v_A_alpha_2 = v_A_alpha_2 * decay

        new_lambda_A = new_v_A + nu_A

        if next_event == EventType.D:
            # Update l_minus
            decay_end = exp(-beta_A * (T_end - next_T))
            update_l_minus = next_alpha_A * (1.0 - decay_end)
            l_minus += update_l_minus
        
            # Update derivatives of l_minus
            l_minus_alpha_0 += alpha_A.partial_0_(next_Q) * (1.0 - decay_end)
            l_minus_alpha_1 += alpha_A.partial_1_(next_Q) * (1.0 - decay_end)
            l_minus_alpha_2 += alpha_A.partial_2_(next_Q) * (1.0 - decay_end)
            l_minus_beta += - update_l_minus / beta_A + update_l_minus * (T_end - next_T)

        else:
            # Update l_plus
            l_plus += log(new_lambda_A)

            # Update derivatives of l_plus
            l_plus_alpha_0 += new_v_A_alpha_0 / new_lambda_A
            l_plus_alpha_1 += new_v_A_alpha_1 / new_lambda_A
            l_plus_alpha_2 += new_v_A_alpha_2 / new_lambda_A
            l_plus_beta += new_v_A_beta / new_lambda_A
            l_plus_nu += 1.0 / new_lambda_A

    # end of for-loop    

    # multiply l_minus and its derivatives wrt alpha and beta by 1/beta_A
    l_minus /= beta_A
    l_minus_alpha_0 /= beta_A
    l_minus_alpha_1 /= beta_A
    l_minus_alpha_2 /= beta_A
    l_minus_beta /= beta_A

    # add first summand to l_minus
    l_minus += nu_A * T_end

    cdef EvalLoglikelihood res
    res.logL = l_plus - l_minus
    res.gradient = vector[double](5, 0.0)
    res.gradient[0] = l_plus_alpha_0 - l_minus_alpha_0
    res.gradient[1] = l_plus_alpha_1 - l_minus_alpha_1
    res.gradient[2] = l_plus_alpha_2 - l_minus_alpha_2
    res.gradient[3] = l_plus_beta - l_minus_beta
    res.gradient[4] = l_plus_nu - l_minus_nu
    return res

def calibration_target_A(
        np.ndarray[double, ndim = 1] params,
        np.ndarray[double, ndim = 1] times,
        np.ndarray[long, ndim = 1] events,
        np.ndarray[long, ndim = 1] states,
        double T_end,
        ):
    cdef double alpha_A_2 = max(params[2], 0.0)
    cdef double beta_A = max(params[3], 0.000001)
    cdef double nu_A = max(params[4], 0.00000001)
    cdef vector[double] time_vector = times 
    cdef vector[EventType] event_vector = events
    cdef vector[long] state_vector = states 
    cdef EvalLoglikelihood res = eval_loglikelihood(
            params[0], params[1], alpha_A_2,
            beta_A, nu_A,
            time_vector,
            event_vector,
            state_vector,
            T_end
            )
    cdef double f =  - res.logL
    cdef np.ndarray[double, ndim=1] grad = nparray(res.gradient, dtype=float)
    cdef np.ndarray[double, ndim=1] g = - grad
    return f, g

def minimize_calibration_target_A(
        np.ndarray[double, ndim=1] times,
        np.ndarray[long, ndim = 1] events,
        np.ndarray[long, ndim=1] states,
        double T_end,
        np.ndarray[double, ndim=1] init_guess, 
        int maxiter = 1000, 
        int disp=0
        ):        
    cdef list bounds = [
            (None, None),
            (None, None),
            (0.0, None),
            (0.000001, None),
            (0.000001, None),
    ]
    res = minimize(
            calibration_target_A,
            init_guess,
            args = (times, events, states, T_end),
            method = 'L-BFGS-B',
            bounds = bounds, 
            jac = True,
            options = {
                'maxiter': maxiter,
                'disp': disp}
            )
    return res


def launch_minimization_A(
        np.ndarray[double, ndim=1] times,
        np.ndarray[long, ndim=1] events,
        np.ndarray[long, ndim=1] states,
        double T_end,
        int num_guesses=5,
        int maxiter=1000, 
        int disp=0,
        launch_async = False
        ):
    cdef list init_guesses = utils.generate_init_guesses(
        times,
        events,
        num=num_guesses,
        )
    cdef list list_args = [
            (times, events, states, T_end, init_guess, maxiter, disp)
            for init_guess in init_guesses
            ]
    cdef list res
    if launch_async:
        res = utils.launch_async(minimize_calibration_target_A, list_args)
    else:    
        res = utils.launch_serial(minimize_calibration_target_A, list_args)
    return res
