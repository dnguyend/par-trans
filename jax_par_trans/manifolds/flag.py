""":math:`Flag`: Flag manifold.
"""
from functools import partial

import jax.numpy as jnp
import jax.numpy.linalg as jla
from jax import jit
from jax.scipy.linalg import expm

from jax_par_trans.expv.utils import (vcat, grand)
from jax_par_trans.expv.expv import LinearOperator


class FlagCanonicalParallelOperator(LinearOperator):
    """ To implment expv of Flag parallel operator
    alpha is .5
    """
    def __init__(self, params):
        self.ar = params['ar']
        self.flag = params['flag']

    def set_params(self, params):
        self.ar = params

    @partial(jit, static_argnums=(0,))
    def dot(self, b):
        ar, salp = self.ar, jnp.sqrt(.5)
        d = ar.shape[1]
        b_a = b[:d, :]
        b_r = b[d:, :]
        a = ar[:d, :]
        r = ar[d:, :]

        return vcat(
            self.flag.proj_m(b_a@a + salp*r.T@b_r),
            (0.5*b_r@a-salp*r@b_a))

    @partial(jit, static_argnums=(0,))
    def one_norm_est(self):
        ar, salp = self.ar, jnp.sqrt(.5)
        d = ar.shape[1]
        na = jnp.max(salp*jnp.sum(jnp.abs(ar[d:, :]), axis=0)
                     + jla.norm(ar[:d, :], 1))
        nr = 0.5*jnp.max(jnp.sum(jnp.abs(ar[:d, :]), axis=0)
                         + 1/salp*jla.norm(ar[d:, :], jnp.inf))

        return jnp.max(jnp.array([na, nr]))


class Flag():
    """:math:`Flag(\\vec{d})` with a homogeneous metric defined by a parameter.
    Realized as a quotient of a Stiefel manifold

    :param alpha: the metric is :math:`tr \\eta^{T}\\eta+(\\alpha-1)tr\\eta^TYY^T\\eta`.
    
    For ease of implementation, :math:`d_{p+1}` is renamed d[0] and saved at top of dvec.
    """
    def __init__(self, dvec, alpha=.5):
        self.n = jnp.sum(dvec)
        self.d = jnp.sum(dvec[:-1])
        self.shape = (self.n, self.d)
        self.alpha = alpha
        self.dvec = jnp.concatenate([dvec[-1:], dvec[:-1]])
        cs = self.dvec[:].cumsum() - self.dvec[0]
        self._g_idx = dict((i+1, (cs[i], cs[i+1]))
                           for i in range(cs.shape[0]-1))
        self.p = self.dvec.shape[0]-1

    def name(self):
        """ name of the object
        """
        return f"Flag({self.dvec})  alpha={self.alpha}"

    @partial(jit, static_argnums=(0,))    
    def symf(self, omg):
        """ symmetrize but keep diagonal blocks unchanged
        """
        p = self.p
        ret = 0.5*(omg+omg.T)
        for tt in range(1, p+1):
            bt, et = self._g_idx[tt]
            ret = ret.at[bt:et, bt:et].set(omg[bt:et, bt:et])
        return ret

    @partial(jit, static_argnums=(0,))
    def proj_m(self, omg):
        """ projection to horizontal space
        """
        p = self.p
        ret = 0.5*(omg-omg.T)
        for tt in range(1, p+1):
            bt, et = self._g_idx[tt]
            ret = ret.at[bt:et, bt:et].set(0.)
        return ret

    def inner(self, x, xi, eta):
        """ Inner product 
        """
        alp = self.alpha
        # ix_xi = x.T@xi
        # ix_eta = x.T@eta
        return jnp.sum(xi*eta) + (alp-1)*jnp.sum((x.T@xi)*(x.T@eta))

    def proj(self, x, omg):
        """ projection to the tangent bundle
        """
        return omg - x@self.symf(x.T@omg)

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
        """  A random vector at x
        """
        tmp, key = self.rand_ambient(key)
        return self.proj(x, tmp), key

    def retract(self, x, v):
        """ second order retraction
        """
        return x + v - 0.5* self.proj(x, self.christoffel_gamma(x, v, v))

    def approx_nearest(self, q):
        """ point on the manifold that is approximately nearest to q
        """
        return jla.qr(q)[0]

    def make_ar(self, a, r):
        """  lift ar a tangent vector to the manifold at :math:`I_{n,d}`
        to a square matrix a horizontal vector at :math:`SO(n)`
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
        """function representing the Christoffel symbols
        """
        alp = self.alpha
        xTxi = x.T@xi
        xTeta = x.T@eta

        def sym2(a):
            return a + a.T

        return x@self.symf(xi.T@eta) - (1-alp)*(
            xi@xTeta + eta@xTxi - x@sym2(xTxi@xTeta))

    def _sc(self, ar, ft):
        """ Scaling the a block of ar by a factor ft
        """
        arn = ar.copy()
        return arn.at[:ar.shape[1], :].set(ar[:ar.shape[1], :]*ft)

    def parallel_canonical(self, x, xi, eta, t):
        """only works for alpha = .5
        parallel transport. Only works for alpha = .5
        The exponential action is computed
        using expv, with our customized estimate of 1_norm of the operator P

        :param x: a point on the manifold
        :param xi: the initial velocity of the geodesic
        :param eta: the vector to be transported
        :param t: time.

        """
        n, d = x.shape
        # alp = 0.5
        salp = jnp.sqrt(self.alpha)
        u, _, _ = jla.svd(xi - x@(x.T@xi), full_matrices=False)
        k = min(n-d, d)
        q = u[:, :k]
        xq = jnp.concatenate([x, q], axis=1)

        ar = xq.T@xi
        # a = ar[:d, :]
        prt0 = xq@expm(t*jnp.concatenate(
            [ar,
             vcat(-ar[d:, :].T, jnp.zeros((k, k)))], axis=1))

        flag_opt = FlagCanonicalParallelOperator({"ar": ar, 'flag': self})

        w = self._sc(flag_opt.expv(self._sc(xq.T@eta, salp), t), 1/salp)

        return prt0@w + (eta - x@x.T@eta - q@q.T@eta)@expm(0.5*t*ar[:d, :])
