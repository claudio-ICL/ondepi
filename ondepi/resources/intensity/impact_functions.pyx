# distutils: language = c++

cdef double phi(double x):
    if x < 0.0:
        return exp(x)
    else:
        return 1.0 + log (1.0 + x)

cdef double phi_prime(double x):
    if x < 0.0:
        return exp(x)
    else:
        return 1.0 / (1.0 + x)

cdef double alpha_D_(double q, double c_0, double c_1, double c_2):
    if (q <= 0):
        return 0
    else:
        return c_2 * phi(c_1 * q + c_0) * exp(-1 / (q * q))

cdef double alpha_D_partial_0(double q, double c_0, double c_1, double c_2):
    if (q <= 0):
      return 0
    else:
      return c_2 * phi_prime(c_1 * q + c_0) * exp(-1 / (q * q))
  
cdef double alpha_D_partial_1(double q, double c_0, double c_1, double c_2):
    if (q <= 0):
      return 0
    else:
      return q * c_2 * phi_prime(c_1 * q + c_0) * exp(-1 / (q * q))

cdef double alpha_D_partial_2(double q, double c_0, double c_1, double c_2):
    if (q <= 0):
        return 0
    else:
        return phi(c_1 * q + c_0) * exp(-1 / (q * q))


cdef class Alpha_D:
    def __cinit__(self, double c_0, double c_1, double c_2):
        self.c_0 = c_0
        self.c_1 = c_1
        self.c_2 = c_2
    cdef double eval_(self, double q):
        return alpha_D_(q, self.c_0, self.c_1, self.c_2)
    cdef double eval__(self, long q):
        return self.eval_(<double>q)
    cdef double partial_0(self, double q):
        return alpha_D_partial_0(q, self.c_0, self.c_1, self.c_2)
    cdef double partial_0_(self, long q):
        return self.partial_0(<double> q)
    cdef double partial_1(self, double q):
        return alpha_D_partial_1(q, self.c_0, self.c_1, self.c_2)
    cdef double partial_1_(self, long q):
        return self.partial_1(<double> q)
    cdef double partial_2(self, double q):
        return alpha_D_partial_2(q, self.c_0, self.c_1, self.c_2)
    cdef double partial_2_(self, long q):
        return self.partial_2(<double> q)


cdef double alpha_A_(double q, double c_0, double c_1, double c_2):
  return c_2 * phi(c_1 * q + c_0);

cdef double alpha_A_partial_0(double q, double c_0, double c_1, double c_2):
  return c_2 * phi_prime(c_1 * q + c_0);

cdef double alpha_A_partial_1(double q, double c_0, double c_1, double c_2):
  return c_2 * phi_prime(c_1 * q + c_0) * q;

cdef double alpha_A_partial_2(double q, double c_0, double c_1, double c_2):
  return phi(c_1 * q + c_0);

cdef class Alpha_A:
    def __cinit__(self, double c_0, double c_1, double c_2):
        self.c_0 = c_0
        self.c_1 = c_1
        self.c_2 = c_2
    cdef double eval_(self, double q):
        return alpha_A_(q, self.c_0, self.c_1, self.c_2)
    cdef double eval__(self, long q):
        return self.eval_(<double>q)
    cdef double partial_0(self, double q):
        return alpha_A_partial_0(q, self.c_0, self.c_1, self.c_2)
    cdef double partial_0_(self, long q):
        return self.partial_0(<double> q)
    cdef double partial_1(self, double q):
        return alpha_A_partial_1(q, self.c_0, self.c_1, self.c_2)
    cdef double partial_1_(self, long q):
        return self.partial_1(<double> q)
    cdef double partial_2(self, double q):
        return alpha_A_partial_2(q, self.c_0, self.c_1, self.c_2)
    cdef double partial_2_(self, long q):
        return self.partial_2(<double> q)
  

cdef double next_v(double v, double T, double next_T, double next_alpha,
              double beta):
  cdef double decay = exp(-beta * (next_T - T));
  return v * decay + next_alpha

cdef double next_v_beta(double v_beta, double v, double T, double next_T,
                   double beta):
  cdef double decay = exp(-beta * (next_T - T))
  return v_beta * decay - v * (next_T - T) * decay

cdef double next_v_alpha(double v_alpha, double T, double next_T,
                    double next_alpha_partial, double beta):
  cdef double decay = exp(-beta * (next_T - T))
  return v_alpha * decay + next_alpha_partial

  
