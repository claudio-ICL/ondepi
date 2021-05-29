import numpy as np
import matplotlib.pyplot as plt


def phi(x):
    if isinstance(x, np.ndarray):
        res = np.zeros_like(x, dtype=np.float)
        idx = x < 0.0
        res[idx] = np.exp(x[idx])
        res[~idx] = 1.0 + np.log(1.0 + x[~idx])
        return res
    if x < 0.0:
        return np.exp(x)
    else:
        return 1.0 + np.log(1.0 + x)


def phi_prime(x):
    if isinstance(x, np.ndarray):
        res = np.zeros_like(x, dtype=np.float)
        idx = x < 0.0
        res[idx] = np.exp(x[idx])
        res[~idx] = 1.0 / (1.0 + x[~idx])
        return res
    if x < 0.0:
        return np.exp(x)
    else:
        return 1.0 / (1.0 + x)


def alpha_D_(q,  c_0,  c_1, c_2):
    if isinstance(q, np.ndarray):
        res = np.zeros_like(q, dtype=np.float)
        idx = q <= 0.0
        res[idx] = 0.0
        res[~idx] = c_2 * phi(c_1 * q[~idx] + c_0) * \
            np.exp(-1 / np.square(q[~idx]))
        return res
    if (q <= 0):
        return 0
    else:
        return c_2 * phi(c_1 * q + c_0) * np.exp(-1 / (q * q))


def alpha_D_partial_0(q,  c_0,  c_1, c_2):
    if (q <= 0):
      return 0
    else:
      return c_2 * phi_prime(c_1 * q + c_0) * np.exp(-1 / (q * q))


def alpha_D_partial_1(q,  c_0,  c_1, c_2):
    if (q <= 0):
      return 0
    else:
      return q * c_2 * phi_prime(c_1 * q + c_0) * np.exp(-1 / (q * q))


def alpha_D_partial_2(q,  c_0,  c_1, c_2):
    if (q <= 0):
      return 0
    else:
      return phi_prime(c_1 * q + c_0) * np.exp(-1 / (q * q))


class Alpha_D:
    def __init__(self,  c_0=10.0,  c_1=-1.0, c_2=1.0):
        self.c_0 = c_0
        self.c_1 = c_1
        self.c_2 = c_2

    def eval_(self,  q):
        return alpha_D_(q, self.c_0, self.c_1, self.c_2)

    def partial_0(self,  q):
        return alpha_D_partial_0(q, self.c_0, self.c_1, self.c_2)

    def partial_1(self,  q):
        return alpha_D_partial_1(q, self.c_0, self.c_1, self.c_2)


def alpha_A_(q,  c_0,  c_1,  c_2):
  return c_2 * phi(c_1 * q + c_0)


def alpha_A_partial_0(q,  c_0,  c_1,  c_2):
  return c_2 * phi_prime(c_1 * q + c_0)


def alpha_A_partial_1(q,  c_0,  c_1,  c_2):
  return c_2 * phi_prime(c_1 * q + c_0) * q


def alpha_A_partial_2(q,  c_0,  c_1,  c_2):
  return phi(c_1 * q + c_0)


class Alpha_A:
    def __init__(self,  c_0=0.0,  c_1=1.0,  c_2=10.0):
        self.c_0 = c_0
        self.c_1 = c_1
        self.c_2 = c_2

    def eval_(self,  q):
        return alpha_A_(q, self.c_0, self.c_1, self.c_2)

    def partial_0(self,  q):
        return alpha_A_partial_0(q, self.c_0, self.c_1, self.c_2)

    def partial_1(self,  q):
        return alpha_A_partial_1(q, self.c_0, self.c_1, self.c_2)

    def partial_2(self,  q):
        return alpha_A_partial_2(q, self.c_0, self.c_1, self.c_2)


def next_v(v,  T,  next_T,  next_alpha,
           beta):
  decay = np.exp(-beta * (next_T - T))
  return v * decay + next_alpha


def next_v_beta(v_beta,  v,  T,  next_T,
                beta):
  decay = np.exp(-beta * (next_T - T))
  return v_beta * decay - v * (next_T - T) * decay


def next_v_alpha(v_alpha,  T,  next_T,
                 next_alpha_partial,  beta):
  decay = np.exp(-beta * (next_T - T))
  return v_alpha * decay + next_alpha_partial


def impact_function(Alpha=Alpha_D, q_max=1000, **kwargs):
    alpha = Alpha(**kwargs)
    qs = np.linspace(0.0, q_max, num=100)
    vals = alpha.eval_(qs)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.plot(qs, vals, label="impact_function")
    return fig
