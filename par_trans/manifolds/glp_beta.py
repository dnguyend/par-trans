""":math:`GL^+`: Positive Component of the Generalized Linear group with a Cheeger deformation metric.
"""
import numpy as np
import numpy.linalg as la
from numpy.random import randn

from par_trans.utils.utils import (sym, asym, lie, cz)
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply, LinearOperator


class GLpBeta():
    """:math:`GL^+` with left invariant metric defined by a parameter.

    :param p: the size of the matrix
    :param beta: the metric is `tr \\mathtt{g}^2 -\frac{2\alpha-1}{2}\\mathtt{g}_{\\mathfrak{a}}^2.

    """
    def __init__(self, n, beta):
        self.shape = (n, n)
        self.beta = beta

    def name(self):
        """ name of the object
        """
        return f"GL+({self.shape[0]})  beta={self.beta}"

    def inner(self, x, xi, eta):
        """ Inner product 
        """
        bt = self.beta
        ix_xi = la.solve(x, xi)
        ix_eta = la.solve(x, eta)
        return np.sum(ix_xi*ix_eta.T) + (1+bt)*np.sum(asym(ix_xi)*asym(ix_eta))

    def proj(self, _, omg):
        return omg

    def rand_ambient(self):
        """random ambient vector
        """
        return randn(*(self.shape))

    def rand_point(self):
        """ A random point on the manifold
        """
        ret = self.rand_ambient()
        if la.det(ret) < 0:
            ret[0, :] = - ret[0, :]
            return ret
        return ret

    def rand_vec(self, _):
        """ A random tangent vector on the manifold
        """
        return self.rand_ambient()

    def retract(self, x, v):
        """ second order retraction, but simple
        """
        return x + v - 0.5* self.proj(x, self.christoffel_gamma(x, v, v))

    def approx_nearest(self, q):
        return q

    def exp(self, x, v):
        """ geodesic, or riemannian exponential
        """
        bt = self.beta
        a = la.solve(x, v)
        return x@expm(a-(1+bt)*asym(a))@expm((1+bt)*asym(a))

    def dexp(self, x, v, t, ddexp=False):
        """ Higher time derivative of geodesics
        return 
        """

        bt = self.beta
        a = la.solve(x, v)

        prt0 = x@expm(t*(a-(1+bt)*asym(a)))
        prt1 = expm(t*((1+bt)*asym(a)))

        if ddexp:
            return prt0@prt1, prt0@a@prt1, \
                prt0@(a@a + (1+bt)*lie(a, asym(a)))@prt1
        return prt0@prt1, prt0@a@prt1

    def christoffel_gamma(self, x, xi, eta):
        """ the Christoffel symbol, as a function
        """
        bt = self.beta
        ix_xi = la.solve(x, xi)
        ix_eta = la.solve(x, eta)
        return -0.5*(xi@la.solve(x, eta) + eta@la.solve(x, xi)) \
            + (bt+1)/2*x@(lie(asym(ix_xi), ix_eta) + lie(asym(ix_eta), ix_xi))

    def parallel(self, x, xi, eta, t):
        n = self.shape[0]
        bt = self.beta
        a = la.solve(x, xi)

        prt0 = x@expm(t*(a-(1+bt)*asym(a)))
        prt1 = expm(t*((1+bt)*asym(a)))

        b = la.solve(x, eta)

        def par(b):
            return 0.5*(lie(b, a) + (1+self.beta)*(lie(asym(a), b) - lie(asym(b), a)))

        def par_T(b):
            return 0.5*(lie(b, a.T) + (1+self.beta)*(lie(-asym(a), b) - asym(lie(b, a.T))))
                
        p_opt = LinearOperator((n**2, n**2),
                               matvec=lambda w: t*par(w.reshape(n, n)).reshape(-1),
                               rmatvec=lambda w: t*par_T(w.reshape(n, n)).reshape(-1))

        return prt0@expm_multiply(p_opt, b.reshape(-1), traceA=0).reshape(n, n)@prt1                           
