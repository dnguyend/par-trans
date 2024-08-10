from time import perf_counter
import matplotlib.pyplot as plt

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
    # c = stf.rand_vec(np.eye(n)[:, :d])

    dlt = 1e-6
    t = .8
    print("CHECK METRIC COMPATIBILITY USING NUMERICAL DERIVATIVE")
    print((stf.inner(x+dlt*v, stf.proj(x+dlt*v, va), stf.proj(x+dlt*v, va)) - stf.inner(x, va, va))/dlt)
    print(2*stf.inner(x, va, (stf.proj(x+dlt*v, va) -va)/dlt + stf.christoffel_gamma(x, v, va)))    
    # print((stf.inner(x+dlt*v, va, va) - stf.inner(x, va, va))/dlt)
    # print(2*stf.inner(x, va, stf.christoffel_gamma(x, v, va)))

    r1 = (stf.proj(x+dlt*v, va) - stf.proj(x, va))/dlt + stf.christoffel_gamma(x, v, va)
    print("Check Christoffel gamma produces a connection")
    print(cz(sym(x.T@r1)))
    
    r1 = stf.exp(x, t*v)
    
    print("Check time derivatives of geodesics")
    print(np.allclose((stf.exp(x, (t+dlt)*v) - stf.exp(x, t*v))/dlt,
                      stf.dexp(x, v, t, ddexp=False)[1], atol=1e-3
                      ))

    print(np.allclose((stf.dexp(x, v, t+dlt)[1] - stf.dexp(x, v, t)[1])/dlt,
                      stf.dexp(x, v, t, ddexp=True)[2], atol=1e-2
                      ))

    gmms = stf.dexp(x, v, t, ddexp=True)

    print("CHECK Geodesic Equation with analytic differentiation")
    print(cz(gmms[2] + stf.christoffel_gamma(gmms[0], gmms[1], gmms[1])))

    t0 = perf_counter()
    Delta1 = stf.parallel_expm_multiply(x, v, va, t)    
    t1 = perf_counter()    
    Delta = stf.parallel(x, v, va, t)
    
    t2 = perf_counter()        
    print(cz(Delta-Delta1))
    print("time expm_multiply =%f(s), time solv_w = %f(s)" % (t1 - t0, t2-t1))
    print("CHECK TRANSPORT EQUATION with numerical differentiation")
    print(cz((stf.parallel(x, v, va, t+dlt) - Delta)/dlt \
          + stf.christoffel_gamma(gmms[0], gmms[1], Delta)))    


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
    print("CHECK PAR_BAL formula")
    print(cz(par_bal(b, ar, salp)
             - sc(par(sc(b, 1/salp), ar, alp), salp)))

    print(np.sum(par_bal(b, ar, salp)*c) +
          np.sum(par_bal(c, ar, salp)*b))
    
    t = 1.2
    # time different way to solve the ODE
    p_opt = ssl.LinearOperator(
        (n*d, n*d),
        matvec=lambda w: t*par(w.reshape(n, d), ar, alp).reshape(-1),
        rmatvec=lambda w: -t*par_T(w.reshape(n, d), ar, alp).reshape(-1))

    gmms = stf.exp(x, t*ar)

    t0 = perf_counter()
    val1 = ssl.expm_multiply(p_opt, b.reshape(-1), traceA=0).reshape(n, d)
    t1 = perf_counter()
    print("Time scipy expm_multiply %f" % (t1 - t0))

    t2 = perf_counter()
    val2 = enp.expm_multiply(p_opt, b.reshape(-1), traceA=0, use_frag_31=False).reshape(n, d)
    t3 = perf_counter()
    print("Time expm_multiply use only 1-norm %f" % (t3 - t2))
    
    t4 = perf_counter()
    val3 = sc(solve_w(sc(b, salp), ar, alp, t), 1/salp)
    t5 = perf_counter()
    print("Time using solve_w %f" % (t5 - t4))
    print("Time ratio between expm_multipy and solve_w %f" % ((t1-t0)/(t5-t4)))
    print("Compare values")
    print(cz(val1-val2))
    print(cz(val1-val3))


def test_isometry():
    np.random.seed(0)
    n = 1000
    d = 200
    alp = 1.
    stf = Stiefel(n, d, alp)
    x = np.concatenate([np.eye(d), np.zeros((n-d, d))])

    n_samples = 20
    all_smpl = []

    def normalize(a, x):
        return a / np.sqrt(stf.inner(x, a, a))

    for _ in range(n_samples):
        spl = stf.rand_vec(x)
        ft = np.random.choice(n_samples, (), replace=True)

        all_smpl.append(ft*normalize(spl, x))

    all_smpl = np.array(all_smpl)


    def cal_cov(gm, smpls):
        mat = np.zeros((n_samples,  n_samples))
        for i in range(n_samples):
            for j in range(i+1):
                mat[i, j] = stf.inner(gm, smpls[i, :, :], smpls[j, :, :])
                if i != j:
                    mat[j, i] = mat[i, j]
        return mat

    cov_0 = cal_cov(x, all_smpl)

    v = stf.rand_vec(x)
    v = v/np.sqrt(stf.inner(x, v, v))

    cov_diff = []
    # t_grid = [0.5, 1., 2., 5., 10., 15.]
    # t_grid = [0.5, 1.]
    t_grid = [0.1, .3, .5, .7, 1.2, 1.5, 1.7, 2.1, 3., 15.]
    for t in t_grid:
        transported = []
        for i in range(n_samples):
            transported.append(stf.parallel(x, v, all_smpl[i, :, :], t))

        transported = np.array(transported)
        gm = stf.exp(x, t*v)
        cov_t = cal_cov(gm, transported)
        # print(cz(cov_t- cov_0))
        cov_diff.append((t, cov_t))

    plt.plot(t_grid, [np.log10(cz(cov_diff[i][1] - cov_0)) for i in range(len(cov_diff))])
    plt.ylim(-14, 0)
    plt.xlabel("t(seconds)")
    plt.ylabel("max log10 of differences")
    plt.title("Maximum absolute difference of inner product matrices")
    plt.savefig("np_cov_diff_stief.png")
    # plt.show()
    display(np.concatenate([np.array(t_grid)[:, None], np.array([np.log10(cz(cov_diff[i][1] - cov_0)) for i in range(len(cov_diff))])[:, None]], axis=1))    


if __name__ == "__main__":
    test_big()    
    test_stiefel_fast()

    test_isometry()
