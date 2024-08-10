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

    print("Check Christoffel gamma produces a metric compatible connection")
    print((soa.inner(x+dlt*v, soa.proj(x+dlt*v, va), soa.proj(x+dlt*v, va)) - soa.inner(x, va, va))/dlt)
    print(2*soa.inner(x, va, soa.christoffel_gamma(x, v, va)))

    print("Check time derivatives of geodesics")
    print(np.allclose((soa.exp(x, (t+dlt)*v) - soa.exp(x, t*v))/dlt,
                      soa.dexp(x, v, t, ddexp=False)[1], atol=1e-3
                      ))

    print(np.allclose((soa.dexp(x, v, t+dlt)[1] - soa.dexp(x, v, t)[1])/dlt,
                      soa.dexp(x, v, t, ddexp=True)[2], atol=1e-2
                      ))

    gmms = soa.dexp(x, v, t, ddexp=True)

    print("CHECK Geodesic Equation with analytic differentiation")    
    print(gmms[2] + soa.christoffel_gamma(gmms[0], gmms[1], gmms[1]))
    
    Delta = soa.parallel(x, v, va, t)
    
    print("CHECK TRANSPORT EQUATION with numerical differentiation")
    print((soa.parallel(x, v, va, t+dlt) - Delta)/dlt \
          + soa.christoffel_gamma(gmms[0], gmms[1], Delta))


if __name__ == "__main__":
    test()
    
