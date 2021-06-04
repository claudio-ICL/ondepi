from numpy import array as nparray
from scipy.optimize import minimize

cdef EvalLoglikelihood eval_loglikelihood(
         double alpha_D_0, double alpha_D_1, double alpha_D_2, 
         double beta_D, double nu_D,
         double alpha_A_0, double alpha_A_1, double alpha_A_2, 
         double beta_A, double nu_A,
         vector[double] times,
         vector[long] queue
         ):
    # Instantiate impact functions
    cdef:
        Alpha_D alpha_D = Alpha_D(alpha_D_0, alpha_D_1, alpha_D_2)
        Alpha_A alpha_A = Alpha_A(alpha_A_0, alpha_A_1, alpha_A_2)

    # Define auxiliary variables
    cdef long unsigned int J = times.size()

    cdef vector[double] lambda_D
    lambda_D.reserve(J)
    cdef vector[double] v_D
    v_D.reserve(J)
    cdef vector[double] v_D_beta
    v_D_beta.reserve(J)
    cdef vector[double] v_D_alpha_0
    v_D_alpha_0.reserve(J)
    cdef vector[double] v_D_alpha_1
    v_D_alpha_1.reserve(J)
    cdef vector[double] v_D_alpha_2
    v_D_alpha_2.reserve(J)

    cdef vector[double] v_A
    v_A.reserve(J)
    cdef vector[double] v_A_beta
    v_A_beta.reserve(J)
    cdef vector[double] v_A_alpha_0
    v_A_alpha_0.reserve(J)
    cdef vector[double] v_A_alpha_1
    v_A_alpha_1.reserve(J)
    cdef vector[double] v_A_alpha_2
    v_A_alpha_2.reserve(J)

    # Set first entries
    v_D.push_back(alpha_D.eval__(queue.front()))
    v_D_beta.push_back(0.0)
    v_D_alpha_0.push_back(alpha_D.partial_0_(queue.front()))
    v_D_alpha_1.push_back(alpha_D.partial_1_(queue.front()))
    v_D_alpha_2.push_back(alpha_D.partial_2_(queue.front()))

    lambda_D.push_back(v_D.front() + nu_D)
  
    v_A.push_back(alpha_A.eval__(queue.front()))
    v_A_beta.push_back(0.0)
    v_A_alpha_0.push_back(alpha_A.partial_0_(queue.front()))
    v_A_alpha_1.push_back(alpha_A.partial_1_(queue.front()))
    v_A_alpha_2.push_back(alpha_A.partial_2_(queue.front()))
  
    cdef:
        double l_plus = log(lambda_D.front())
        double l_D_minus = 0.0
        double l_A_minus = 0.0
      
        double l_plus_alpha_0 = v_D_alpha_0.front() / lambda_D.front()
        double l_plus_alpha_1 = v_D_alpha_1.front() / lambda_D.front()
        double l_plus_alpha_2 = v_D_alpha_2.front() / lambda_D.front()
        double l_plus_beta = v_D_beta.front() / lambda_D.front()
        double l_plus_nu = 1.0 / lambda_D.front()
      
        double l_D_minus_alpha_0 = 0.0
        double l_D_minus_alpha_1 = 0.0
        double l_D_minus_alpha_2 = 0.0
        double l_D_minus_beta = 0.0
        double l_D_minus_nu = 0.0
      
        double l_A_minus_alpha_0 = 0.0
        double l_A_minus_alpha_1 = 0.0
        double l_A_minus_alpha_2 = 0.0
        double l_A_minus_beta = 0.0
        double l_A_minus_nu = 0.0
  
    cdef:
        double T
        double next_T
        long Q
        long next_Q
        double next_alpha_D
        double next_alpha_A
        double decay_D
        double decay_A
        long unsigned int j

    for j in range(J - 1):        
        T = times[j]
        next_T = times[j + 1]
        Q = queue[j]
        next_Q = queue[j + 1]
        next_alpha_D = alpha_D.eval__(next_Q)
        next_alpha_A = alpha_A.eval__(next_Q)
        v_D.push_back(next_v(v_D[j], T, next_T, next_alpha_D, beta_D))
        v_D_beta.push_back(next_v_beta(v_D_beta[j], v_D[j], T, next_T, beta_D))
        v_D_alpha_0.push_back(next_v_alpha(
            v_D_alpha_0[j], T, next_T,
            alpha_D.partial_0_(next_Q), beta_D
            )
        )    
        v_D_alpha_1.push_back(next_v_alpha(
            v_D_alpha_1[j], T, next_T,
            alpha_D.partial_1_(next_Q), beta_D
            )
        )    
        v_D_alpha_2.push_back(next_v_alpha(
            v_D_alpha_2[j], T, next_T,
            alpha_D.partial_2_(next_Q), beta_D
            )
        )    

        lambda_D.push_back(v_D.back() + nu_D)
    
        v_A.push_back(next_v(v_A[j], T, next_T, next_alpha_A, beta_A))
        v_A_beta.push_back(next_v_beta(v_A_beta[j], v_A[j], T, next_T, beta_A))
        v_A_alpha_0.push_back(next_v_alpha(
            v_A_alpha_0[j], T, next_T,
            alpha_A.partial_0_(next_Q), beta_A
            )
        )    
        v_A_alpha_1.push_back(next_v_alpha(
            v_A_alpha_1[j], T, next_T,
            alpha_A.partial_1_(next_Q), beta_A
            )
        )    
        v_A_alpha_2.push_back(next_v_alpha(
            v_A_alpha_2[j], T, next_T,
            alpha_A.partial_2_(next_Q), beta_A
            )
        )    
    
        l_plus += log(lambda_D.back())
        l_D_minus += next_alpha_D - v_D.back()
        l_A_minus += next_alpha_A - v_A.back()
    
        l_plus_alpha_0 += v_D_alpha_0.back() / lambda_D.back()
        l_plus_alpha_1 += v_D_alpha_1.back() / lambda_D.back()
        l_plus_alpha_2 += v_D_alpha_2.back() / lambda_D.back()
        l_plus_beta += v_D_beta.back() / lambda_D.back()
        l_plus_nu += 1.0 / lambda_D.back()
    
        decay_D = exp(-beta_D * (next_T - T))
        l_D_minus_alpha_0 -= v_D_alpha_0[j] * decay_D
        l_D_minus_alpha_1 -= v_D_alpha_1[j] * decay_D
        l_D_minus_alpha_2 -= v_D_alpha_2[j] * decay_D
        l_D_minus_beta -= (next_alpha_D - v_D.back()) / beta_D + v_D_beta.back()
    
        decay_A = exp(-beta_A * (next_T - T))
        l_A_minus_alpha_0 -= v_A_alpha_0[j] * decay_A
        l_A_minus_alpha_1 -= v_A_alpha_1[j] * decay_A
        l_A_minus_alpha_2 -= v_A_alpha_2[j] * decay_A
        l_A_minus_beta -= (next_alpha_A - v_A.back()) + v_A_beta.back()

  
    l_D_minus /= beta_D
    l_D_minus += nu_D * times.back()
    l_A_minus /= beta_A
    l_A_minus += nu_A * times.back()
  
    l_D_minus_alpha_0 /= beta_D
    l_D_minus_alpha_1 /= beta_D
    l_D_minus_alpha_2 /= beta_D
    l_D_minus_beta /= beta_D
    l_D_minus_nu = times.back()
  
    l_A_minus_alpha_0 /= beta_A
    l_A_minus_alpha_1 /= beta_A
    l_A_minus_alpha_2 /= beta_A
    l_A_minus_beta /= beta_A
    l_A_minus_nu = times.back()
  
    cdef EvalLoglikelihood res
    res.logL = l_plus - l_D_minus - l_A_minus
    res.gradient = vector[double](10, 0.0)
    res.gradient[0] = l_plus_alpha_0 - l_D_minus_alpha_0
    res.gradient[1] = l_plus_alpha_1 - l_D_minus_alpha_1
    res.gradient[2] = l_plus_alpha_2 - l_D_minus_alpha_2
    res.gradient[3] = l_plus_beta - l_D_minus_beta
    res.gradient[4] = l_plus_nu - l_D_minus_nu
    res.gradient[5] = -l_A_minus_alpha_0
    res.gradient[6] = -l_A_minus_alpha_1
    res.gradient[7] = -l_A_minus_alpha_2
    res.gradient[8] = -l_A_minus_beta
    res.gradient[9] = -l_A_minus_nu
    return res



def calibration_target(
        np.ndarray[double, ndim = 1] params,
        np.ndarray[double, ndim = 1] times,
        np.ndarray[long, ndim = 1] states,
        ):
    cdef double alpha_D_2 = max(params[2], 0.0)
    cdef double beta_D = max(params[3], 0.000001)
    cdef double nu_D = max(params[4], 0.00000001)
    cdef double alpha_A_2 = max(params[7], 0.0)
    cdef double beta_A = max(params[8], 0.000001)
    cdef double nu_A = max(params[9], 0.0)
    cdef vector[double] time_vector = times 
    cdef vector[long] state_vector = states 
    cdef EvalLoglikelihood res = eval_loglikelihood(
            params[0], params[1], alpha_D_2,
            beta_D, nu_D,
            params[5], params[6], alpha_A_2,
            beta_A, nu_A,
            time_vector,
            state_vector
            )
    cdef double f =  - res.logL
    cdef np.ndarray[double, ndim=1] grad = nparray(res.gradient, dtype=float)
    cdef np.ndarray[double, ndim=1] g = - grad
    return f, g


def minimize_calibration_target(
        np.ndarray[double, ndim=1] times,
        np.ndarray[long, ndim=1] states,
        int maxiter = 100000, 
        int disp=0
        ):        
    cdef np.ndarray[double, ndim=1] x0 = nparray([
            1.0, 0.5, 1.0,
            10.0, 10.0,
            10.0, -2.0, 1.0,
            10.0, 10.0,
            ], dtype=float)
    cdef list bounds = [
            (None, None),
            (None, None),
            (0.0, None),
            (0.000001, None),
            (0.000001, None),
            (None, None),
            (None, None),
            (0.0, None),
            (0.000001, None),
            (0.000001, None),
            ]
            
    res = minimize(
            calibration_target,
            x0,
            args = (times, states),
            method = 'L-BFGS-B',
            bounds = bounds, 
            jac = True,
            options = {
                'maxiter': maxiter,
                'disp': disp}
            )
    return res
