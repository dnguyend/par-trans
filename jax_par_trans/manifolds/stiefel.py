""":math:`St`: Stiefel manifold.
"""
from functools import partial

import jax.numpy as jnp
import jax.numpy.linalg as jla
from jax import jit
from jax.scipy.linalg import expm

from ..expv.utils import (sym, asym, vcat, grand)
from ..expv.expv import LinearOperator


class StiefelParallelOperator(LinearOperator):
    """ Defining the operator P used in parallel transport
    on Stiefel manifolds
    """
    def __init__(self, params):
        self.ar = params['ar']
        self.salp = jnp.sqrt(params['alpha'])

    def set_params(self, params):
        self.ar = params['ar']
        if 'alpha' in params:
            self.salp = jnp.sqrt(params['alpha'])

    @partial(jit, static_argnums=(0,))
    def dot(self, b):
        ar, salp = self.ar, self.salp
        d = ar.shape[1]
        b_a = b[:d, :]
        b_r = b[d:, :]
        a = ar[:d, :]
        r = ar[d:, :]

        return vcat(
            ((4*salp**2-1)*asym(b_a@a) + salp*asym(r.T@b_r)),
            (salp**2*b_r@a-salp*r@b_a))

    @partial(jit, static_argnums=(0,))
    def one_norm_est(self):
        ar, salp = self.ar, self.salp
        d = ar.shape[1]
        na = salp*jnp.max(jnp.sum(jnp.abs(ar[d:, :]), axis=0)
                          + jnp.abs(4*salp**2-1)/salp*jla.norm(ar[:d, :], 1))
        nr = salp**2*jnp.max(jnp.sum(jnp.abs(ar[:d, :]), axis=0)
                             + 1/salp*jla.norm(ar[d:, :], jnp.inf))

        return jnp.max(jnp.array([na, nr]))


class StiefelOperator(LinearOperator):
    """ Evaluating expm([[a, -r.T], [r, 0]])
    Operate on a thin vector.
    For testing. 
    """
    def __init__(self, params):
        self.ar = params

    def set_params(self, params):
        self.ar = params

    @partial(jit, static_argnums=(0,))
    def dot(self, b):
        d = self.ar.shape[1]
        return vcat(self.ar[:d, :]@b[:d, :] - self.ar[d:, :].T@b[d:, :],
                    self.ar[d:, :]@b[:d, :])

    def one_norm_est(self):
        return jla.norm(self.ar, jnp.inf)



class Stiefel():
    """:math:`\\mathrm{St}_{n,d}` with an invariant metric defined by a parameter.

    :param p: the size of the matrix
    :param alpha: the metric is :math:`tr \\eta^{T}\\eta+(\\alpha-1)tr\\eta^TYY^T\\eta`.
    """
    def __init__(self, n, d, alpha):
        self.shape = (n, d)
        self.alpha = alpha
        self.d = d

    def name(self):
        """ name of the object
        """
        return f"Stiefel({self.shape})  alpha={self.alpha}"

    def inner(self, x, xi, eta):
        """ Inner product 
        """
        alp = self.alpha
        return jnp.sum(xi*eta) + (alp-1)*jnp.sum((x.T@xi)*(x.T@eta))

    def proj(self, x, omg):
        return omg - x@sym(x.T@omg)

    def rand_ambient(self, key):
        """random ambient vector
        """
        return grand(key, self.shape)

    def rand_point(self, key):
        """ A random point on the manifold
        """
        tmp, key = self.rand_ambient(key)
        return jla.qr(tmp)[0], key

    def rand_vec(self, key, x):
        """ A random tangent vector to the manifold at x
        """
        tmp, key = self.rand_ambient(key)
        return self.proj(x, tmp), key

    def retract(self, x, v):
        """ second order retraction.
        """
        return x + v - 0.5* self.proj(x, self.christoffel_gamma(x, v, v))

    def approx_nearest(self, q):
        """ point on the manifold that is approximately nearest to q
        """
        return jla.qr(q)[0]

    def make_ar(self, a, r):
        """  lift ar a tangent vector to the manifold at :math:`I_{n,d}`
        to a square matrix, the lifted horizontal vector at :math:`I_n\\in SO(n)`.
        """        
        k = r.shape[0]
        return jnp.concatenate([
            jnp.concatenate([a, - r.T], axis=1),
            jnp.concatenate([r, jnp.zeros((k, k))], axis=1)], axis=0)

    def exp(self, x, v):
        """ geodesic, or riemannian exponential
        """
        n, d = x.shape
        u, _, _ = jla.svd(v - x@(x.T@v), full_matrices=False)
        k = min(n-d, d)
        q = u[:, :k]
        a = x.T@v
        r = q.T@v

        aar = self.make_ar(2*self.alpha*a, r)
        return (jnp.concatenate([x, q], axis=1)@expm(aar)[:, :d])@expm((1-2*self.alpha)*a)

    def dexp(self, x, v, t, ddexp=False):
        """ Higher derivative of Exponential function.

        :param x: the initial point :math:`\\gamma(0)`
        :param v: the initial velocity :math:`\\dot{\\gamma}(0)`
        :param t: time.

        If ddexp is False, we return :math:`\\gamma(t), \\dot{\\gamma}(t)`.
        Otherwise, we return :math:`\\gamma(t), \\dot{\\gamma}(t), \\ddot{\\gamma}(t)`.
        """
        n, d = x.shape
        alp = self.alpha
        u, _, _ = jla.svd(v - x@(x.T@v), full_matrices=False)
        k = jnp.min(jnp.array([n-d, d]))
        q = u[:, :k]
        a = x.T@v
        r = q.T@v

        ar = self.make_ar(a, r)
        aar = self.make_ar(2*alp*a, r)
        prt0 = jnp.concatenate([x, q], axis=1)@expm(t*aar)
        prt1 = expm(t*(1-2*self.alpha)*a)
        if not ddexp:
            return prt0[:, :d]@prt1, (prt0@ar)[:, :d]@prt1

        lie_ar_a0 = jnp.zeros_like(ar)
        lie_ar_a0 = lie_ar_a0.at[d:, :d].set(ar[d:, :d]@a)
        lie_ar_a0 = lie_ar_a0.at[:d, d:].set(- lie_ar_a0[d:, :d].T)

        return prt0[:, :d]@prt1, \
            (prt0@ar)[:, :d]@prt1, \
            (prt0@(ar@ar + (1-2*alp)*lie_ar_a0))[:, :d]@prt1

    def christoffel_gamma(self, x, xi, eta):
        """Christoffel function of the manifold
        """
        alp = self.alpha
        xTxi = x.T@xi
        xTeta = x.T@eta

        def sym2(a):
            return a + a.T

        return 0.5*x@(xi.T@eta + eta.T@xi) - (1-alp)*(
            xi@xTeta + eta@xTxi - x@sym2(xTxi@xTeta))

    def _sc(self, ar, ft):
        """ Scaling the a block of ar by a factor ft
        """
        arn = ar.copy()
        return arn.at[:ar.shape[1], :].set(ar[:ar.shape[1], :]*ft)

    def parallel(self, x, xi, eta, t):
        """parallel transport. The exponential action is computed
        using expv, with our customized estimate of 1_norm of the operator P

        :param x: a point on the manifold
        :param xi: the initial velocity of the geodesic
        :param eta: the vector to be transported
        :param t: time.
        """
        
        n, d = x.shape
        alp = self.alpha
        salp = jnp.sqrt(self.alpha)
        u, _, _ = jla.svd(xi - x@(x.T@xi), full_matrices=False)
        k = jnp.min(jnp.array([n-d, d]))
        q = u[:, :k]
        xq = jnp.concatenate([x, q], axis=1)

        ar = xq.T@xi
        a = ar[:d, :]
        r = ar[d:, :]

        aar = jnp.concatenate([vcat(2*alp*a, r),
                               vcat(-r.T, jnp.zeros((k, k)))], axis=1)
        prt0 = xq@expm(t*aar)
        prt1 = expm(t*(1-2*alp)*a)
        sp_opt = StiefelParallelOperator({"ar": ar, "alpha": alp})

        w = self._sc(sp_opt.expv(self._sc(xq.T@eta, salp), t), 1/salp)

        return prt0@w@prt1 \
            + (eta - x@x.T@eta - q@q.T@eta)@expm(t*(1-alp)*a)
