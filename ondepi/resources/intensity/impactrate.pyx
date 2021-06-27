cdef double compute_decay(double time, double previous_time, double beta):
    return exp(- beta * (time - previous_time))

cdef double compute_next_v(double v, double T, double next_T, double beta):
    cdef double decay = compute_decay(next_T, T, beta)
    return 1.0 + v * decay

cdef double compute_next_v_beta(double v, double v_beta, double T, double next_T, double beta):
    cdef double decay = compute_decay(next_T, T, beta)
    cdef double dT = next_T - T
    return v_beta * decay - v * decay * dT

cdef class ImpactRate:
    def __cinit__(self, double alpha, double beta):
        self.alpha = max(0.0, alpha)
        self.beta = max(0.0000000001, beta)
        cdef ImpactRateValue value
        value.time=0.0
        value.value=0.0
        self.value = value
    cdef double next_value(self, double next_T):
        cdef double T = self.value.time
        cdef double previous_value = self.value.value
        cdef double decay = compute_decay(next_T, T, self.beta)
        return self.alpha + previous_value * decay
    cdef void innovate(self, double next_T):
        self.value.value = self.next_value(next_T)
        self.value.time = next_T
    cdef double compute_at_arrival(self, double next_T):
        self.innovate(next_T)
        return self.value.value
    cdef ImpactRateValue get_value(self):
        return self.value
    cdef double eval_after_last_event(self, double time):
        cdef double decay = compute_decay(time, self.value.time, self.beta)
        cdef double value_at_last_event = self.value.value
        return value_at_last_event * decay





