"""define a class of linear operators with an exponential action method.
"""
from functools import partial

import jax.numpy as jnp
from jax import jit, lax


# The first 30 values are from table A.3 of Computing Matrix Functions.
# other are from various table
_theta_array = jnp.array([
    [1, 2.29e-16],
    [2, 2.58e-8],
    [3, 1.39e-5],
    [4, 3.40e-4],
    [5, 2.40e-3],
    [6, 9.07e-3],
    [7, 2.38e-2],
    [8, 5.00e-2],
    [9, 8.96e-2],
    [10, 1.44e-1],
    [11, 2.14e-1],
    [12, 3.00e-1],
    [13, 4.00e-1],
    [14, 5.14e-1],
    [15, 6.41e-1],
    [16, 7.81e-1],
    [17, 9.31e-1],
    [18, 1.09],
    [19, 1.26],
    [20, 1.44],
    [21, 1.62],
    [22, 1.82],
    [23, 2.01],
    [24, 2.22],
    [25, 2.43],
    [26, 2.64],
    [27, 2.86],
    [28, 3.08],
    [29, 3.31],
    [30, 3.54],
    # The rest are from table 3.1 of]
    # Computing the Action of the Matrix Exponential.]
    [35, 4.7],
    [40, 6.0],
    [45, 7.2],
    [50, 8.5],
    [55, 9.9]])



class LinearOperator():
    """ A class of linear operators with method for
    exponential action. The operator operates on a vector space via dot.
    the exponential action expv(t*self, b) is provided through the method expv.
    To use the operator, define a derived class of linear operator supplying an estimate for 1-norm.
    
    """
    def __init__(self, params=None):
        pass

    def set_params(self, params):
        """Override the params supplied in constructor
        this is to avoid creating new object repeatedly.
        """
        raise NotImplementedError

    def dot(self, b):
        """Let the operator operates on b
        """
        raise NotImplementedError

    def one_norm_est(self):
        """Estimate of the 1-norm
        """
        raise NotImplementedError

    @partial(jit, static_argnums=(0, 3))
    def expv(self, b, t, tol=None):
        """Exponential action on b. Return is exp(t self)b
        """
        norm_est = t*self.one_norm_est()
        agm = jnp.argmin(jnp.ceil(norm_est/_theta_array[:, 1])*_theta_array[:, 0])
        m_star, s = _theta_array[agm, 0], jnp.ceil(norm_est/_theta_array[agm, 1])

        if tol is None:
            u_d = 2 ** -53
            tol = u_d

        b = b.copy()
        f = b.copy()

        def norm2(x):
            return jnp.sqrt(jnp.sum(x*x))

        def body_fun(carry):
            # b is the power sequence, f is the sum of sequence
            j, b, f, pass_norm, current_norm = carry
            pass_norm = current_norm

            b = t / (s*(j+1)) * self.dot(b)
            f = f + b
            current_norm = norm2(b)
            return j+1, b, f, pass_norm, current_norm

        def cond_fun(carry):
            j, _, f, pass_norm, current_norm = carry
            return (j < m_star) & (pass_norm + current_norm > tol * norm2(f))

        def func(j, carry):
            j, b, f, pass_norm, current_norm = carry
            j, b, f, pass_norm, current_norm = lax.while_loop(cond_fun, body_fun, (0, b, f, jnp.inf, norm2(b)))
            return j, f, f, pass_norm, current_norm

        ret = lax.fori_loop(lower=0, upper=s.astype(int), body_fun=func,
                            init_val=(0, b, f, jnp.inf, jnp.inf))

        return ret[2]
