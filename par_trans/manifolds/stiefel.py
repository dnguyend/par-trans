"""Stiefel manifold :math:`\\mathrm{St}((n, d), \\alpha)` with metric defined by a parameters :math:`\\alpha`.
"""


import numpy as np
import numpy.linalg as la
from numpy.random import randn
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator, expm_multiply

from par_trans.utils.utils import (sym, asym, vcat)


def par_bal(b, ar, salp):
    """ balanced parallel operator
    :math:`s_{salp}(P(ar, s_{\\frac{1}{salp}}(b)))`, where s is the operator scaling the top
    :math:`d\\times d` block in a :math:`n\\times d` matrix by the subscript argument.
    salp is typically :math:`\\alpha^{\\frac{1}{2}}`.
    This operator is antisymmetric when restricted to the tangent space at :math:`I_{n,d}`.
    """
    d = ar.shape[1]
    b_a = b[:d, :]
    b_r = b[d:, :]
    a = ar[:d, :]
    r = ar[d:, :]

    return vcat(
        ((4*salp**2-1)*asym(b_a@a) + salp*asym(r.T@b_r)),
        (salp**2*b_r@a-salp*r@b_a))


def solve_w(b, ar, alp, t, tol=None):
    """The exponential action :math:`expv(tP_{ar}, b)` when the metric is
    given by the parameter :math:`\\alpha`.
    The calculation uses the  1-norm estimate in the local function one_norm_est.
    """
    _theta = {
        # The first 30 values are from table A.3 of Computing Matrix Functions.
        1: 2.29e-16,
        2: 2.58e-8,
        3: 1.39e-5,
        4: 3.40e-4,
        5: 2.40e-3,
        6: 9.07e-3,
        7: 2.38e-2,
        8: 5.00e-2,
        9: 8.96e-2,
        10: 1.44e-1,
        # 11
        11: 2.14e-1,
        12: 3.00e-1,
        13: 4.00e-1,
        14: 5.14e-1,
        15: 6.41e-1,
        16: 7.81e-1,
        17: 9.31e-1,
        18: 1.09,
        19: 1.26,
        20: 1.44,
        # 21
        21: 1.62,
        22: 1.82,
        23: 2.01,
        24: 2.22,
        25: 2.43,
        26: 2.64,
        27: 2.86,
        28: 3.08,
        29: 3.31,
        30: 3.54,
        # The rest are from table 3.1 of
        # Computing the Action of the Matrix Exponential.
        35: 4.7,
        40: 6.0,
        45: 7.2,
        50: 8.5,
        55: 9.9,
    }

    salp = np.sqrt(alp)
    _, d = ar.shape

    def one_norm_est():
        na = t*salp*la.norm(
            np.concatenate([
                ar[d:, :],  np.abs(4*alp-1)/salp*la.norm(ar[:d, :],1)*np.ones((1, d))]), 1)

        nr = t*salp*(salp*la.norm(np.concatenate(
            [ar[:d, :], 1/salp*la.norm(ar[d:, :], np.inf)*np.ones((1, d))]),
                                  1))
        return max(na, nr)

    norm_est = one_norm_est()

    def calc_m_s(norm_est):
        best_m = None
        best_s = None
        for m, theta in _theta.items():
            s = int(np.ceil(norm_est / theta))
            if best_m is None or m * s < best_m * best_s:
                best_m = m
                best_s = s
        return best_m, best_s

    m_star, s = calc_m_s(norm_est)

    if tol is None:
        u_d = 2 ** -53
        tol = u_d
    f = b.copy()

    def norm2(x):
        return np.sqrt(np.sum(x*x))

    for _ in range(s):
        c1 = norm2(b)
        for j in range(m_star):
            b = t / float(s*(j+1)) * par_bal(b, ar, salp)
            c2 = norm2(b)
            f = f + b
            if c1 + c2 <= tol * norm2(f):
                break
            c1 = c2
        b = f
    return f


class Stiefel():
    """:math:`\\mathrm{St}_{n,d}` with an invariant metric defined by a parameter.

    :param p: the size of the matrix
    :param alpha: the metric is :math:`tr \\eta^{T}\\eta+(\\alpha-1)tr\\eta^TYY^T\\eta`.
    """
    def __init__(self, n, d, alpha, null_cut_off=1e-12):
        self.shape = (n, d)
        self.alpha = alpha
        self.d = d
        self.null_cut_off = null_cut_off

    def name(self):
        """ name of the object
        """
        return f"Stiefel({self.shape})  alpha={self.alpha}"

    def inner(self, x, xi, eta):
        """ Inner product 
        """
        alp = self.alpha
        # ix_xi = x.T@xi
        # ix_eta = x.T@eta
        return np.sum(xi*eta) + (alp-1)*np.sum((x.T@xi)*(x.T@eta))

    def proj(self, x, omg):
        return omg - x@sym(x.T@omg)

    def rand_ambient(self):
        """random ambient vector
        """
        return randn(*(self.shape))

    def rand_point(self):
        """ A random point on the manifold
        """
        return la.qr(self.rand_ambient())[0]

    def rand_vec(self, x):
        """ A random tangent vector to the manifold at x
        """
        return self.proj(x, self.rand_ambient())

    def retract(self, x, v):
        """ second order retraction.
        """
        return x + v - 0.5* self.proj(x, self.christoffel_gamma(x, v, v))

    def approx_nearest(self, q):
        """ point on the manifold that is approximately nearest to q
        """        
        return la.qr(q)[0]

    def make_ar(self, a, r):
        """  lift ar a tangent vector to the manifold at :math:`I_{n,d}`
        to a square matrix, the lifted horizontal vector at :math:`I_n\\in SO(n)`.
        """
        k = r.shape[0]
        return np.concatenate([
            np.concatenate([a, - r.T], axis=1),
            np.concatenate([r, np.zeros((k, k))], axis=1)], axis=0)

    def exp(self, x, v):
        """ geodesic, or riemannian exponential
        """
        n, d = x.shape
        u, _, _ = la.svd(v - x@(x.T@v), full_matrices=False)
        k = min(n-d, d)
        q = u[:, :k]
        a = x.T@v
        r = q.T@v

        aar = self.make_ar(2*self.alpha*a, r)
        return (np.concatenate([x, q], axis=1)@expm(aar)[:, :d])@expm((1-2*self.alpha)*a)
    
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
        u, _, _ = la.svd(v - x@(x.T@v), full_matrices=False)
        k = min(n-d, d)
        q = u[:, :k]
        a = x.T@v
        r = q.T@v

        ar = self.make_ar(a, r)
        aar = self.make_ar(2*alp*a, r)
        prt0 = np.concatenate([x, q], axis=1)@expm(t*aar)
        prt1 = expm(t*(1-2*self.alpha)*a)
        if not ddexp:
            return prt0[:, :d]@prt1, (prt0@ar)[:, :d]@prt1
        
        lie_ar_a0 = np.zeros_like(ar)
        lie_ar_a0[d:, :d] = ar[d:, :d]@a
        lie_ar_a0[:d, d:] = - lie_ar_a0[d:, :d].T

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

        return 0.5*x@(xi.T@eta + eta.T@xi) - (1-alp)*(
            xi@xTeta + eta@xTxi - x@sym2(xTxi@xTeta))

    def parallel_expm_multiply(self, x, xi, eta, t):
        """parallel transport.  The exponential action is computed using expm_multiply from scipy.

        :param x: a point on the manifold
        :param xi: the initial velocity of the geodesic
        :param eta: the vector to be transported
        :param t: time.

        """
        n, d = x.shape
        alp = self.alpha
        u, _, _ = la.svd(xi - x@(x.T@xi), full_matrices=False)
        k = min(n-d, d)
        q = u[:, :k]
        a = x.T@xi
        r = q.T@xi

        # ar = self.make_ar(a, r)
        xq = np.concatenate([x, q], axis=1)
        aar = self.make_ar(2*alp*a, r)
        prt0 = xq@expm(t*aar)
        prt1 = expm(t*(1-2*self.alpha)*a)

        def par(b):
            b_a = b[:d, :]
            b_r = b[d:, :]
            return vcat(
                (4*alp-1)*asym(b_a@a) + asym(r.T@b_r),
                alp*(b_r@a-r@b_a))
        # return 0.5*(lie(b, a) + (1-2*alp)*(lie(a_k_0, b) - lie(self._lie_proj_a(b), a)))

        def par_T(b):
            b_a = b[:d, :]
            b_r = b[d:, :]

            return vcat(
                -(4*alp-1)*asym(b_a)@a - alp*r.T@b_r,
                r@asym(b_a) - alp*b_r@a)

        p_opt = LinearOperator(((d+k)*d, (d+k)*d),
                               matvec=lambda w: t*par(w.reshape(d+k, d)).reshape(-1),
                               rmatvec=lambda w: t*par_T(w.reshape(d+k, d)).reshape(-1))

        return prt0@expm_multiply(p_opt, (xq.T@eta).reshape(-1), traceA=0).reshape(d+k, d)@prt1 \
            + (eta - x@x.T@eta - q@q.T@eta)@expm(t*(1-alp)*a)

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
        u, _, _ = la.svd(xi - x@(x.T@xi), full_matrices=False)
        k = min(n-d, d)
        q = u[:, :k]
        xq = np.concatenate([x, q], axis=1)
        
        ar = xq.T@xi
        a = ar[:d, :]
        r = ar[d:, :]

        # ar = self.make_ar(a, r)
        xq = np.concatenate([x, q], axis=1)
        aar = self.make_ar(2*alp*a, r)
        prt0 = xq@expm(t*aar)
        prt1 = expm(t*(1-2*self.alpha)*a)

        def sc(ar, ft):
            """ Scaling the a block of ar by a factor ft
            """
            arn = ar.copy()
            arn[:ar.shape[1], :] = ar[:ar.shape[1], :]*ft
            return arn

        # w = sc(sp_opt.expv(sc(xq.T@eta, salp), t), 1/salp)
        salp = np.sqrt(alp)
        w = sc(solve_w(sc(xq.T@eta, salp), ar, alp, t), 1/salp)

        return prt0@w@prt1 + (eta - x@x.T@eta - q@q.T@eta)@expm(t*(1-alp)*a)
