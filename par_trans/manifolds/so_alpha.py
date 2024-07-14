""":math:`SO`: Special Orthogonal group with a Cheeger deformation metric.
"""
import numpy as np
import numpy.linalg as la
from numpy.random import randn
from par_trans.utils.utils import (sym, asym, lie)
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply, LinearOperator


class SOAlpha():
    """:math:`SO` with left invariant metric defined by a parameter.

    :param n: the size of the matrix
    :param alpha: the metric is `\\frac{1}{2}tr \\mathtt{g}^2 -\frac{2\alpha-1}{2}\\mathtt{g}_{\\mathfrak{a}}^2.

    """
    def __init__(self, n, k, alpha):
        self.shape = (n, n)
        self.alpha = alpha
        self.k = k

    def name(self):
        """ name of the object
        """
        return f"SO({self.shape[0]})  alpha={self.alpha}"

    def inner(self, x, xi, eta):
        """ Inner product 
        """
        alp, k = self.alpha, self.k
        ix_xi = x.T@xi
        ix_eta = x.T@eta
        return 0.5*np.sum(xi*eta) + (alp-.5)*np.sum(ix_xi[:k, :k]*ix_eta[:k, :k])

    def proj(self, x, omg):
        return x@asym(x.T@omg)

    def _lie_proj_a(self, omg):
        k = self.k
        ret = np.zeros_like(omg)
        ret[:k, :k] = omg[:k, :k]
        return ret

    def rand_ambient(self):
        """random ambient vector
        """
        return randn(*(self.shape))

    def rand_point(self):
        """ A random point on the manifold
        """
        return la.qr(self.rand_ambient())[0]
        

    def rand_vec(self, x):
        """ A random point on the manifold
        """
        return self.proj(x, self.rand_ambient())

    def retract(self, x, v):
        """ second order retraction, but simple
        """
        return x + v - 0.5* self.proj(x, self.christoffel_gamma(x, v, v))

    def approx_nearest(self, q):
        return la.qr(q)[0]

    def exp(self, x, v):
        """ geodesic, or riemannian exponential
        """
        alp, k = self.alpha, self.k
        a = x.T@v
        a_k = a[:k, :k]
        a_alp = a.copy()
        a_alp[:k, :k] = 2*alp*a_k
        xe_a_alp = x@expm(a_alp)
        xe_a_alp[:, :k] = xe_a_alp[:, :k]@expm((1-2*alp)*a_k)
        return xe_a_alp

    def dexp(self, x, v, t, ddexp=False):
        """ Higher derivative of Exponential function
        """
        alp, k = self.alpha, self.k
        a = x.T@v
        a_k = a[:k, :k]
        a_alp = a.copy()
        a_alp[:k, :k] = 2*alp*a_k
        xe_a_alp_tmp = x@expm(t*a_alp)
        e_a_k = expm(t*(1-2*alp)*a_k)
        
        xe_a_alp = np.concatenate([xe_a_alp_tmp[:, :k]@e_a_k,
                                   xe_a_alp_tmp[:, k:]], axis=1)

        dxe_a_alp = xe_a_alp_tmp@a
        dxe_a_alp[:, :k] = dxe_a_alp[:, :k]@e_a_k

        if not ddexp:
            return xe_a_alp, dxe_a_alp

        lie_a_a_k = np.zeros(x.shape)
        lie_a_a_k[k:, :k] = a[k:, :k]@a_k
        lie_a_a_k[:k, k:] = -lie_a_a_k[k:, :k].T

        ddxe_a_alp = xe_a_alp_tmp@(a@a + (1-2*alp)*lie_a_a_k)
        ddxe_a_alp[:, :k] = ddxe_a_alp[:, :k]@e_a_k

        return xe_a_alp, dxe_a_alp, ddxe_a_alp

    def christoffel_gamma_lie(self, x, xi, eta):
        alp, k = self.alpha, self.k
        ix_xi = x.T@xi
        ix_eta = x.T@eta
        ix_xi0 = np.zeros(x.shape)
        ix_eta0 = np.zeros(x.shape)
        ix_xi0[:k, :k] = ix_xi[:k, :k]
        ix_eta0[:k, :k] = ix_eta[:k, :k]

        return 0.5*x@(xi.T@eta + eta.T@xi) \
            + (1-2*alp)/2*x@(lie(ix_xi0, ix_eta) + lie(ix_eta0, ix_xi))

    def christoffel_gamma(self, x, xi, eta):
        alp, k = self.alpha, self.k
        ix_xi = x.T@xi
        ix_eta = x.T@eta

        lie2 = np.zeros_like(x)
        lie2[k:, :k] = - ix_xi[k:, :k]@ix_eta[:k, :k] - ix_eta[k:, :k]@ix_xi[:k, :k]
        lie2[:k, k:] = -lie2[k:, :k].T

        return 0.5*x@(xi.T@eta + eta.T@xi) + (1-2*alp)/2*x@lie2

    def parallel(self, x, xi, eta, t):
        alp, n, k = self.alpha, self.shape[0], self.k
        a = x.T@xi
        a_k = a[:k, :k]
        a_alp = a.copy()
        a_alp[:k, :k] = 2*alp*a_k
        xe_a_alp = x@expm(t*a_alp)
        e_a_k = expm(t*(1-2*alp)*a_k)

        b = x.T@eta

        a_k_0 = np.zeros_like(x)
        a_k_0[:k, :k] = a_k

        def par(b):
            return 0.5*(lie(b, a) + (1-2*alp)*(lie(a_k_0, b) - lie(self._lie_proj_a(b), a)))

        def par_T(b):
            return 0.5*(lie(b, a.T) + (1-2*alp)*(lie(-a_k_0, b) - self._lie_proj_a((lie(b, a.T)))))

        p_opt = LinearOperator((n**2, n**2),
                               matvec=lambda w: t*par(w.reshape(n, n)).reshape(-1),
                               rmatvec=lambda w: t*par_T(w.reshape(n, n)).reshape(-1))
        ccc = expm_multiply(p_opt, b.reshape(-1), traceA=0).reshape(n, n)

        return xe_a_alp@np.concatenate([ccc[:, :k]@e_a_k, ccc[:, k:]], axis=1)

    
