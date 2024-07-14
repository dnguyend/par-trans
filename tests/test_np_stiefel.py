import numpy as np
import numpy.linalg as la
from numpy.random import randn
from scipy.linalg import expm
import scipy.sparse.linalg as ssl

from par_trans.manifolds import Stiefel
from par_trans.utils.utils import (sym, asym, cz, vcat)
import par_trans.utils.expm_multiply_np as enp



def test_big():
    n = 300
    d = 200
    alp = .6
    stf = Stiefel(n, d, alp)

    x = stf.rand_point()
    v = stf.rand_vec(x)
    va = stf.rand_vec(x)
    c = stf.rand_vec(np.eye(n)[:, :d])

    dlt = 1e-6
    t = .8

    print((stf.inner(x+dlt*v, va, va) - stf.inner(x, va, va))/dlt)
    print(2*stf.inner(x, va, stf.christoffel_gamma(x, v, va)))

    r1 = (stf.proj(x+dlt*v, va) - stf.proj(x, va))/dlt + stf.christoffel_gamma(x, v, va)
    print(cz(sym(x.T@r1)))
    
    r1 = stf.exp(x, t*v)
    
    print(np.allclose((stf.exp(x, (t+dlt)*v) - stf.exp(x, t*v))/dlt,
                      stf.dexp(x, v, t, ddexp=False)[1], atol=1e-3
                      ))

    print(np.allclose((stf.dexp(x, v, t+dlt)[1] - stf.dexp(x, v, t)[1])/dlt,
                      stf.dexp(x, v, t, ddexp=True)[2], atol=1e-2
                      ))

    gmms = stf.dexp(x, v, t, ddexp=True)
    print(cz(gmms[2] + stf.christoffel_gamma(gmms[0], gmms[1], gmms[1])))
    
    Delta = stf.parallel(x, v, va, t)
    Delta1 = stf.parallel_expm_multiply(x, v, va, t)
    
    print((stf.parallel(x, v, va, t+dlt) - Delta)/dlt \
          + stf.christoffel_gamma(gmms[0], gmms[1], Delta))


def sc(ar, ft):
    arn = ar.copy()
    arn[:ar.shape[1], :] *= ft
    return arn


def par(b, ar, alp):
    d = ar.shape[1]
    b_a = b[:d, :]
    b_r = b[d:, :]
    a = ar[:d, :]
    r = ar[d:, :]

    return vcat(
        (4*alp-1)*asym(b_a@a) + asym(r.T@b_r),
        alp*(b_r@a-r@b_a))


def par_T(b, ar, alp):
    d = ar.shape[1]
    b_a = b[:d, :]
    b_r = b[d:, :]
    a = ar[:d, :]
    r = ar[d:, :]

    return vcat(
        -(4*alp-1)*asym(b_a)@a - alp*r.T@b_r,
        r@asym(b_a) - alp*b_r@a)


def par_bal(b, ar, salp):
    d = ar.shape[1]
    b_a = b[:d, :]
    b_r = b[d:, :]
    a = ar[:d, :]
    r = ar[d:, :]

    return vcat(
        ((4*salp**2-1)*asym(b_a@a) + salp*asym(r.T@b_r)),
        (salp**2*b_r@a-salp*r@b_a))


def solve_w(b, ar, alp, t, tol=None):
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

    
def test_stiefel_fast():
    n = 1000
    d = 500
    
    alp = .6
    stf = Stiefel(n, d, alp)
    x = np.concatenate([np.eye(d), np.zeros((n-d, d))])

    def normalize(x, ar):
        return ar/np.sqrt(stf.inner(x, ar, ar))
        
    ar = stf.rand_vec(x)    
    b = stf.rand_vec(x)
    c = stf.rand_vec(x)

    ar = normalize(x, ar)
    b = normalize(x, b)
    c = normalize(x, c)

    salp = np.sqrt(alp)
    # par_bal
    print(cz(par_bal(b, ar, salp)
             - sc(par(sc(b, 1/salp), ar, alp), salp)))
    
    t = 1.2
    # time different way to solve the ODE
    p_opt = ssl.LinearOperator(
        (n*d, n*d),
        matvec=lambda w: t*par(w.reshape(n, d), ar, alp).reshape(-1),
        rmatvec=lambda w: -t*par_T(w.reshape(n, d), ar, alp).reshape(-1))

    gmms = stf.exp(x, t*ar)

    from time import perf_counter

    t0 = perf_counter()
    val1 = ssl.expm_multiply(p_opt, b.reshape(-1), traceA=0).reshape(n, d)
    t1 = perf_counter()
    print(t1 - t0)

    t2 = perf_counter()
    val2 = enp.expm_multiply(p_opt, b.reshape(-1), traceA=0, use_frag_31=False).reshape(n, d)
    t3 = perf_counter()
    print(t3 - t2)
    
    t4 = perf_counter()
    val3 = sc(solve_w(sc(b, salp), ar, alp, t), 1/salp)
    t5 = perf_counter()
    print(t5 - t4)
    
    print(cz(val1-val2))
    print(cz(val1-val3))
