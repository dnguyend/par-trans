import numpy as np
import numpy.linalg as la
from numpy.random import randn

from par_trans.manifolds import SOAlpha
from par_trans.utils.utils import (sym, asym, lie, cz)

from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply, LinearOperator


def test():
    n = 5
    k = 3
    alp = .6
    soa = SOAlpha(n, k, alp)

    x = soa.rand_point()
    v = soa.rand_vec(x)
    va = soa.rand_vec(x)
    c = soa.rand_vec(np.eye(n))    

    dlt = 1e-6
    t = .8

    print((soa.inner(x+dlt*v, va, va) - soa.inner(x, va, va))/dlt)
    print(2*soa.inner(x, va, soa.christoffel_gamma(x, v, va)))    
    
    print(np.allclose((soa.exp(x, (t+dlt)*v) - soa.exp(x, t*v))/dlt,
                      soa.dexp(x, v, t, ddexp=False)[1], atol=1e-3
                      ))

    print(np.allclose((soa.dexp(x, v, t+dlt)[1] - soa.dexp(x, v, t)[1])/dlt,
                      soa.dexp(x, v, t, ddexp=True)[2], atol=1e-2
                      ))

    gmms = soa.dexp(x, v, t, ddexp=True)
    print(gmms[2] + soa.christoffel_gamma(gmms[0], gmms[1], gmms[1]))
    
    def parallel_dev(self, x, xi, eta, t):
        alp, n = self.alpha, self.shape[0]
        a = x.T@xi
        a_k = a[:k, :k]
        a_alp = a.copy()
        a_alp[:k, :k] = 2*alp*a_k
        xe_a_alp_tmp = x@expm(t*a_alp)
        e_a_k = expm(t*(1-2*alp)*a_k)

        xe_a_alp = xe_a_alp_tmp.copy()
        xe_a_alp[:, :k] = xe_a_alp[:, :k]@e_a_k

        b = x.T@eta

        a_k_0 = np.zeros_like(x)
        a_k_0[:k, :k] = a_k

        # b_k_0 = np.zeros_like(x)
        # b_k_0[:k, :k] = b[:k, :k]

        def par(b):
            return 0.5*(lie(b, a) + (1-2*alp)*(lie(a_k_0, b) - lie(self._lie_proj_a(b), a)))
        # print(par(b) - par1(b))
        
        def par_T(b):
            return 0.5*(lie(b, a.T) + (1-2*alp)*(lie(-self._lie_proj_a(a), b) - self._lie_proj_a((lie(b, a.T)))))

        print(np.sum(c*par(b)) - np.sum(b*par_T(c)))
                        

        p_opt = LinearOperator((n**2, n**2),
                               matvec=lambda w: t*par(w.reshape(n, n)).reshape(-1),
                               rmatvec=lambda w: t*par_T(w.reshape(n, n)).reshape(-1))
        ccc = expm_multiply(p_opt, b.reshape(-1), traceA=0).reshape(n, n)
    
        return xe_a_alp_tmp@np.concatenate([ccc[:, :k]@e_a_k, ccc[:, k:]], axis=1)

    Delta = soa.parallel(x, v, va, t)

    print((soa.parallel(x, v, va, t+dlt) - Delta)/dlt \
          + soa.christoffel_gamma(gmms[0], gmms[1], Delta))
