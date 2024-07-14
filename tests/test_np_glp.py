import numpy as np
import numpy.linalg as la
from numpy.random import randn

from par_trans.manifolds import GLpBeta
from par_trans.utils.utils import (sym, asym, lie, cz)

from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply, LinearOperator

    
def test():
    np.random.seed(1)
    n = 5
    beta = 1.2
    glp = GLpBeta(n, beta)

    x = glp.rand_point()
    v = glp.rand_vec(x)
    va = glp.rand_vec(x)
    vb = glp.rand_vec(x)

    t = .8

    dlt = 1e-6
    
    print((glp.inner(x+dlt*v, va, va) - glp.inner(x, va, va))/dlt)
    print(2*glp.inner(x, va, glp.christoffel_gamma(x, v, va)))
    
    
    print(cz((glp.exp(x, (t+dlt)*v) - glp.exp(x, t*v))/dlt
             - glp.dexp(x, v, t, ddexp=False)[1]))

    print(cz((glp.dexp(x, v, t+dlt)[1] - glp.dexp(x, v, t)[1])/dlt
             - glp.dexp(x, v, t, ddexp=True)[2]))

    gmms = glp.dexp(x, v, t, ddexp=True)
    print(cz(gmms[2] + glp.christoffel_gamma(gmms[0], gmms[1], gmms[1])))

    def Par(b, a):
        return 0.5*(lie(b, a) + (1+glp.beta)*(lie(asym(a), b) - lie(asym(b), a)))

    def Par_T(b, a):
        return 0.5*(lie(b, a.T) + (1+glp.beta)*(lie(-asym(a), b) - asym(lie(b, a.T))))

    print(np.sum(Par(va, v)*vb))
    print(np.sum(Par_T(vb, v)*va))

    def sc(a, ft):
        return sym(a) + ft*asym(a)

    def par_bal(b, a):
        bnew = sc(b, 1/np.sqrt(glp.beta))
        return sc(0.5*(lie(bnew, a) + (1+glp.beta)*(lie(asym(a), bnew) - lie(asym(bnew), a))),
                  np.sqrt(glp.beta))


    def par_bal(b, a):
        bnew = sc(b, 1/np.sqrt(glp.beta))
        return sc(0.5*(lie(bnew, a) + (1+glp.beta)*(lie(asym(a), bnew) - lie(asym(bnew), a))),
                  np.sqrt(glp.beta))
    
    print(np.sum(sc(Par(sc(vb, 1/np.sqrt(glp.beta)), v), np.sqrt(glp.beta))*va))
    print(np.sum(sc(Par(sc(va, 1/np.sqrt(glp.beta)), v), np.sqrt(glp.beta))*vb))

    Delta = glp.parallel(x, v, va, t)

    print((glp.parallel(x, v, va, t+dlt) - Delta)/dlt \
          + glp.christoffel_gamma(gmms[0], gmms[1], Delta))

