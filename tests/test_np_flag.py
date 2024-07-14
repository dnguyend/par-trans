import numpy as np
import numpy.linalg as la
from numpy.random import randn

from par_trans.manifolds import Flag
from par_trans.utils.utils import (sym, asym, lie, cz)

from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply, LinearOperator


def test():
    dvec = np.array([5, 2, 3])
    alp = .5
    flg = Flag(dvec, alp)

    x = flg.rand_point()
    v = flg.rand_vec(x)
    va = flg.rand_vec(x)
    c = flg.rand_vec(np.eye(flg.n)[:, :flg.d])

    dlt = 1e-6
    t = .8

    print((flg.inner(x+dlt*v, va, va) - flg.inner(x, va, va))/dlt)
    print(2*flg.inner(x, va, flg.christoffel_gamma(x, v, va)))

    r1 = (flg.proj(x+dlt*v, va) - flg.proj(x, va))/dlt + flg.christoffel_gamma(x, v, va)
    print(sym(x.T@r1))
    
    r1 = flg.exp(x, t*v)
    
    print(np.allclose((flg.exp(x, (t+dlt)*v) - flg.exp(x, t*v))/dlt,
                      flg.dexp(x, v, t, ddexp=False)[1], atol=1e-3
                      ))

    print(np.allclose((flg.dexp(x, v, t+dlt)[1] - flg.dexp(x, v, t)[1])/dlt,
                      flg.dexp(x, v, t, ddexp=True)[2], atol=1e-2
                      ))

    gmms = flg.dexp(x, v, t, ddexp=True)
    print(cz(gmms[2] + flg.christoffel_gamma(gmms[0], gmms[1], gmms[1])))

    Delta = flg.parallel_canonical(x, v, va, t)
    print(cz(flg.symf(gmms[0].T@Delta)))

    Delta1 = flg.parallel_canonical_expm_multiply(x, v, va, t)
    print(cz(Delta-Delta1))

    print((flg.parallel_canonical(x, v, va, t+dlt) - Delta)/dlt \
          + flg.christoffel_gamma(gmms[0], gmms[1], Delta))
