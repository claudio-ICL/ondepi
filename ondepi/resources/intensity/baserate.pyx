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

cdef double nu(double q, double c_0, double c_1, double c_2):
  return c_2 * phi(c_1 * q + c_0);

cdef double nu_partial_0(double q, double c_0, double c_1, double c_2):
  return c_2 * phi_prime(c_1 * q + c_0);

cdef double nu_partial_1(double q, double c_0, double c_1, double c_2):
  return c_2 * phi_prime(c_1 * q + c_0) * q;

cdef double nu_partial_2(double q, double c_0, double c_1, double c_2):
  return phi(c_1 * q + c_0);

cdef class BaseRate:
    def __cinit__(self, double c_0, double c_1, double c_2):
        self.c_0 = c_0
        self.c_1 = c_1
        self.c_2 = c_2
    cdef double compute_at_arrival(self, long new_state):
        self.set_state(new_state)
        return self.get_value()
    cdef void set_state(self, long q):
        self.state = q
        self.value = self.eval__(q)
    cdef double get_value(self):
        return self.value
    cdef double eval_(self, double q):
        return nu(q, self.c_0, self.c_1, self.c_2)
    cdef double eval__(self, long q):
        return self.eval_(<double>q)
    cdef double partial_0(self, double q):
        return nu_partial_0(q, self.c_0, self.c_1, self.c_2)
    cdef double partial_0_(self, long q):
        return self.partial_0(<double> q)
    cdef double get_partial_0(self):
        return self.partial_0_(self.state)
    cdef double partial_1(self, double q):
        return nu_partial_1(q, self.c_0, self.c_1, self.c_2)
    cdef double partial_1_(self, long q):
        return self.partial_1(<double> q)
    cdef double get_partial_1(self):
        return self.partial_1_(self.state)
    cdef double partial_2(self, double q):
        return nu_partial_2(q, self.c_0, self.c_1, self.c_2)
    cdef double partial_2_(self, long q):
        return self.partial_2(<double> q)
    cdef double get_partial_2(self):
        return self.partial_2_(self.state)
