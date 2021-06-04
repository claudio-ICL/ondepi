from numpy import array as nparray
from scipy.optimize import minimize

cdef EvalLoglikelihood eval_loglikelihood(
         double alpha_D_0, double alpha_D_1, double alpha_D_2, 
         double beta_D, double nu_D,
         vector[double] times, # increasing sequence of times T^D_j when events D happened
         vector[long] states, # size of the queue at T^D_j
         double T_end # length of the observation period, assumed to start at 0.0
         ):
    # Instantiate impact functions
    cdef Alpha_D alpha_D = Alpha_D(alpha_D_0, alpha_D_1, alpha_D_2)

    # Define auxiliary variables
    cdef long unsigned int J = times.size()

    cdef double v_D
    cdef double v_D_alpha_0
    cdef double v_D_alpha_1
    cdef double v_D_alpha_2
    cdef double v_D_beta
    cdef double lambda_D

    # Set first entries
    v_D = alpha_D.eval__(states.front())
    v_D_alpha_0 = alpha_D.partial_0_(states.front())
    v_D_alpha_1 = alpha_D.partial_1_(states.front())
    v_D_alpha_2 = alpha_D.partial_2_(states.front())
    v_D_beta = 0.0

    lambda_D = v_D + nu_D
  
    cdef:
        double l_plus = log(lambda_D)
        double decay_end = exp(-beta_D * (T_end - times.front()))
        double l_minus = alpha_D.eval__(states.front()) * (1.0 - decay_end)
        # The muliplication by 1/beta_D will be done at the end
      
        double l_plus_alpha_0 = v_D_alpha_0 / lambda_D
        double l_plus_alpha_1 = v_D_alpha_1 / lambda_D
        double l_plus_alpha_2 = v_D_alpha_2 / lambda_D
        double l_plus_beta = v_D_beta / lambda_D
        double l_plus_nu = 1.0 / lambda_D

        double l_minus_alpha_0 = alpha_D.partial_0_(states.front()) * (1.0 - decay_end)
        double l_minus_alpha_1 = alpha_D.partial_1_(states.front()) * (1.0 - decay_end)
        double l_minus_alpha_2 = alpha_D.partial_2_(states.front()) * (1.0 - decay_end)
        double l_minus_beta = - l_minus / beta_D + l_minus * (T_end - times.front())
        double l_minus_nu = T_end
  
    cdef:
        double new_v_D = v_D
        double new_v_D_alpha_0 = v_D_alpha_0
        double new_v_D_alpha_1 = v_D_alpha_1
        double new_v_D_alpha_2 = v_D_alpha_2
        double new_v_D_beta = v_D_beta
        double new_lambda_D = lambda_D
        double T
        double next_T
        long Q
        long next_Q
        double next_alpha_D
        double update_l_minus
        long unsigned int j

    for j in range(J - 1):        
        T = times[j]
        next_T = times[j + 1]
        Q = states[j]
        next_Q = states[j + 1]
        next_alpha_D = alpha_D.eval__(next_Q)
        v_D = new_v_D
        v_D_alpha_0 = new_v_D_alpha_0
        v_D_alpha_1 = new_v_D_alpha_1
        v_D_alpha_2 = new_v_D_alpha_2
        v_D_beta = new_v_D_beta
        lambda_D = new_lambda_D
        new_v_D = next_v(v_D, T, next_T, next_alpha_D, beta_D)
        new_v_D_beta = next_v_beta(v_D_beta, v_D, T, next_T, beta_D)
        new_v_D_alpha_0 = next_v_alpha(
            v_D_alpha_0, T, next_T,
            alpha_D.partial_0_(next_Q), beta_D
            )
        new_v_D_alpha_1 = next_v_alpha(
            v_D_alpha_1, T, next_T,
            alpha_D.partial_1_(next_Q), beta_D
            )
        new_v_D_alpha_2 = next_v_alpha(
            v_D_alpha_2, T, next_T,
            alpha_D.partial_2_(next_Q), beta_D
            )

        new_lambda_D = new_v_D + nu_D
    
        # Update l_plus and l_minus
        l_plus += log(new_lambda_D)
        decay_end = exp(-beta_D * (T_end - next_T))
        update_l_minus = next_alpha_D * (1.0 - decay_end)
        l_minus += update_l_minus

        # Update derivatives of l_plus
        l_plus_alpha_0 += new_v_D_alpha_0 / new_lambda_D
        l_plus_alpha_1 += new_v_D_alpha_1 / new_lambda_D
        l_plus_alpha_2 += new_v_D_alpha_2 / new_lambda_D
        l_plus_beta += new_v_D_beta / new_lambda_D
        l_plus_nu += 1.0 / new_lambda_D
    
        # Update derivatives of l_minus
        l_minus_alpha_0 += alpha_D.partial_0_(next_Q) * (1.0 - decay_end)
        l_minus_alpha_1 += alpha_D.partial_1_(next_Q) * (1.0 - decay_end)
        l_minus_alpha_2 += alpha_D.partial_2_(next_Q) * (1.0 - decay_end)
        l_minus_beta += - update_l_minus / beta_D + update_l_minus * (T_end - next_T)

    # end of for-loop    

    # multiply l_minus and its derivatives wrt alpha and beta by 1/beta_D
    l_minus /= beta_D
    l_minus_alpha_0 /= beta_D
    l_minus_alpha_1 /= beta_D
    l_minus_alpha_2 /= beta_D
    l_minus_beta /= beta_D

    # add first summand to l_minus
    l_minus += nu_D * T_end

    cdef EvalLoglikelihood res
    res.logL = l_plus - l_minus
    res.gradient = vector[double](5, 0.0)
    res.gradient[0] = l_plus_alpha_0 - l_minus_alpha_0
    res.gradient[1] = l_plus_alpha_1 - l_minus_alpha_1
    res.gradient[2] = l_plus_alpha_2 - l_minus_alpha_2
    res.gradient[3] = l_plus_beta - l_minus_beta
    res.gradient[4] = l_plus_nu - l_minus_nu
    return res


def calibration_target_D(
        np.ndarray[double, ndim = 1] params,
        np.ndarray[double, ndim = 1] times,
        np.ndarray[long, ndim = 1] states,
        double T_end,
        ):
    cdef double alpha_D_2 = max(params[2], 0.0)
    cdef double beta_D = max(params[3], 0.000001)
    cdef double nu_D = max(params[4], 0.00000001)
    cdef vector[double] time_vector = times 
    cdef vector[long] state_vector = states 
    cdef EvalLoglikelihood res = eval_loglikelihood(
            params[0], params[1], alpha_D_2,
            beta_D, nu_D,
            time_vector,
            state_vector,
            T_end
            )
    cdef double f =  - res.logL
    cdef np.ndarray[double, ndim=1] grad = nparray(res.gradient, dtype=float)
    cdef np.ndarray[double, ndim=1] g = - grad
    return f, g


def minimize_calibration_target_D(
        np.ndarray[double, ndim=1] times,
        np.ndarray[long, ndim=1] states,
        double T_end,
        int maxiter = 100000, 
        int disp=0
        ):        
    cdef np.ndarray[double, ndim=1] x0 = nparray([
            1.0, 0.5, 1.0,
            10.0, 10.0,
            ], dtype=float)
    cdef list bounds = [
            (None, None),
            (None, None),
            (0.0, None),
            (0.000001, None),
            (0.000001, None),
            ]
            
    res = minimize(
            calibration_target_D,
            x0,
            args = (times, states, T_end),
            method = 'L-BFGS-B',
            bounds = bounds, 
            jac = True,
            options = {
                'maxiter': maxiter,
                'disp': disp}
            )
    return res
