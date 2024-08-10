from time import perf_counter
import matplotlib.pyplot as plt

import numpy as np

from par_trans.manifolds import Flag
from par_trans.utils.utils import (sym, cz)


def test_sanity():
    dvec = np.array([50, 20, 30, 400])
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
    print("Check Christoffel gamma produces a metric compatible connection")
    print(cz(sym(x.T@r1)))
    print(cz(flg.symf(x.T@r1)))
    
    r1 = flg.exp(x, t*v)
    
    print("Check time derivatives of geodesics")    
    print(np.allclose((flg.exp(x, (t+dlt)*v) - flg.exp(x, t*v))/dlt,
                      flg.dexp(x, v, t, ddexp=False)[1], atol=1e-3
                      ))

    print(np.allclose((flg.dexp(x, v, t+dlt)[1] - flg.dexp(x, v, t)[1])/dlt,
                      flg.dexp(x, v, t, ddexp=True)[2], atol=1e-2
                      ))

    gmms = flg.dexp(x, v, t, ddexp=True)
    
    print("CHECK Geodesic Equation with analytic differentiation")    
    print(cz(gmms[2] + flg.christoffel_gamma(gmms[0], gmms[1], gmms[1])))

    t0 = perf_counter()
    Delta1 = flg.parallel_canonical_expm_multiply(x, v, va, t)
    t1 = perf_counter()    
    Delta = flg.parallel_canonical(x, v, va, t)
    
    t2 = perf_counter()        
    print(cz(Delta-Delta1))
    print("time expm_multiply =%f(s), time solv_w = %f(s)" % (t1 - t0, t2-t1))    

    print("CHECK TRANSPORT EQUATION with numerical differentiation")
    print(cz((flg.parallel_canonical(x, v, va, t+dlt) - Delta)/dlt \
             + flg.christoffel_gamma(gmms[0], gmms[1], Delta)))


def test_isometry():
    np.random.seed(0)
    n = 1000
    dvec = np.array([50, 20, 30, 900])

    alp = .5
    flg = Flag(dvec, alp)
    x = np.concatenate([np.eye(flg.shape[1]),
                        np.zeros((n-flg.shape[1], flg.shape[1]))])

    n_samples = 20
    all_smpl = []

    def normalize(a, x):
        return a / np.sqrt(flg.inner(x, a, a))

    for _ in range(n_samples):
        spl = flg.rand_vec(x)
        ft = np.random.choice(n_samples, (), replace=True)

        all_smpl.append(ft*normalize(spl, x))

    all_smpl = np.array(all_smpl)


    def cal_cov(gm, smpls):
        mat = np.zeros((n_samples,  n_samples))
        for i in range(n_samples):
            for j in range(i+1):
                mat[i, j] = flg.inner(gm, smpls[i, :, :], smpls[j, :, :])
                if i != j:
                    mat[j, i] = mat[i, j]
        return mat

    cov_0 = cal_cov(x, all_smpl)

    v = flg.rand_vec(x)
    v = v/np.sqrt(flg.inner(x, v, v))

    cov_diff = []
    # t_grid = [0.5, 1., 2., 5., 10., 15.]
    # t_grid = [0.5, 1.]
    t_grid = [0.1, .3, .5, .7, 1.2, 1.5, 1.7, 2.1, 3., 15.]
    for t in t_grid:
        transported = []
        for i in range(n_samples):
            transported.append(flg.parallel_canonical(x, v, all_smpl[i, :, :], t))

        transported = np.array(transported)
        gm = flg.exp(x, t*v)
        cov_t = cal_cov(gm, transported)
        # print(cz(cov_t- cov_0))
        cov_diff.append((t, cov_t))

    plt.plot(t_grid, [np.log10(cz(cov_diff[i][1] - cov_0)) for i in range(len(cov_diff))])
    plt.ylim(-14, 0)
    plt.xlabel("t(seconds)")
    plt.ylabel("max log10 of differences")
    plt.title("Maximum absolute difference of inner product matrices")
    plt.savefig("np_cov_diff_flag.png")
    # plt.show()
    print(np.concatenate([np.array(t_grid)[:, None], np.array([np.log10(cz(cov_diff[i][1] - cov_0)) for i in range(len(cov_diff))])[:, None]], axis=1))
    

if __name__ == "__main__":
    test_sanity()
    test_isometry()
    
