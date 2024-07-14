from time import perf_counter
import timeit
import pandas as  pd

import jax
import jax.numpy as jnp
from jax import jvp, random
from jax_par_trans.expv.utils import (cz, sym)
from jax_par_trans.manifolds import Stiefel


def test():    
    jax.config.update("jax_enable_x64", True)   
    n = 5
    d = 3
    alp = .6
    stf = Stiefel(n, d, alp)

    key = random.PRNGKey(0)

    x, key = stf.rand_point(key)
    v, key = stf.rand_vec(key, x)
    va, key = stf.rand_vec(key, x)
    c, key = stf.rand_vec(key, jnp.eye(n)[:, :d])

    dlt = 1e-6
    t = .8

    print((stf.inner(x+dlt*v, va, va) - stf.inner(x, va, va))/dlt)
    print(2*stf.inner(x, va, stf.christoffel_gamma(x, v, va)))

    r1 = jvp(lambda t: stf.proj(x+t*v, va), (0.,), (1.,))[1] + stf.christoffel_gamma(x, v, va)
    print(sym(x.T@r1))
    
    r1 = stf.exp(x, t*v)
    
    print(cz(jvp(lambda t: stf.exp(x, t*v), (t,), (1.,))[1]
             - stf.dexp(x, v, t, ddexp=False)[1]))

    print(cz(jvp(lambda t: stf.dexp(x, v, t)[1], (t,), (1.,))[1]
             - stf.dexp(x, v, t, ddexp=True)[2]))

    gmms = stf.dexp(x, v, t, ddexp=True)
    print(cz(gmms[2] + stf.christoffel_gamma(gmms[0], gmms[1], gmms[1])))
    
    Delta = stf.parallel(x, v, va, t)

    print((stf.parallel(x, v, va, t+dlt) - Delta)/dlt \
          + stf.christoffel_gamma(gmms[0], gmms[1], Delta))

    print(jvp(lambda t: stf.parallel(x, v, va, t), (t,), (1.,))[1] \
          + stf.christoffel_gamma(gmms[0], gmms[1], Delta))


def test_big():    
    jax.config.update("jax_enable_x64", True)   
    n = 400
    d = 200
    alp = .6
    stf = Stiefel(n, d, alp)

    key = random.PRNGKey(0)

    x, key = stf.rand_point(key)
    v, key = stf.rand_vec(key, x)
    va, key = stf.rand_vec(key, x)
    c, key = stf.rand_vec(key, jnp.eye(n)[:, :d])

    dlt = 1e-6
    t = .8

    print((stf.inner(x+dlt*v, va, va) - stf.inner(x, va, va))/dlt)
    print(2*stf.inner(x, va, stf.christoffel_gamma(x, v, va)))

    r1 = jvp(lambda t: stf.proj(x+t*v, va), (0.,), (1.,))[1] + stf.christoffel_gamma(x, v, va)
    print(sym(x.T@r1))
    
    r1 = stf.exp(x, t*v)
    
    print(cz(jvp(lambda t: stf.exp(x, t*v), (t,), (1.,))[1]
             - stf.dexp(x, v, t, ddexp=False)[1]))

    print(cz(jvp(lambda t: stf.dexp(x, v, t)[1], (t,), (1.,))[1]
             - stf.dexp(x, v, t, ddexp=True)[2]))

    gmms = stf.dexp(x, v, t, ddexp=True)
    print(cz(gmms[2] + stf.christoffel_gamma(gmms[0], gmms[1], gmms[1])))
    
    Delta = stf.parallel(x, v, va, t)

    print((stf.parallel(x, v, va, t+dlt) - Delta)/dlt \
          + stf.christoffel_gamma(gmms[0], gmms[1], Delta))

    print(cz(jvp(lambda t: stf.parallel(x, v, va, t), (t,), (1.,))[1] \
          + stf.christoffel_gamma(gmms[0], gmms[1], Delta)))


def test_one_set(stf, key, t_interval, n_samples=10, n_repeats=5):
    ret = []
    d = stf.shape[1]
    for _ in range(n_samples):
        ret_spl = []
        x, key = stf.rand_point(key)
        v, key = stf.rand_vec(key, x)
        va, key = stf.rand_vec(key, x)
        # compile the git
        stf.dexp(x, v, 1.)
        par = stf.parallel(x, v, va, 1.)

        for t in t_interval:
            ret_t = []
            for _ in range(n_repeats):
                t0 = perf_counter()
                gmms = stf.dexp(x, v, t)
                t1 = perf_counter()
                t_gmms = t1 - t0
                
                t3 = perf_counter()
                par = stf.parallel(x, v, va, t)
                t4 = perf_counter()
                t_par = t4 - t3

                # check accuracy:
                geo_man = cz(gmms[0].T@gmms[0] - jnp.eye(d))
                par_tan = cz(sym(gmms[0].T@par))
                par_eq = cz(jvp(lambda t: stf.parallel(x, v, va, t), (t,), (1.,))[1] +
                            stf.christoffel_gamma(gmms[0], gmms[1], par))
                
                ret_t.append([t_gmms, t_par, geo_man, par_tan, par_eq])
                
                
            ret_spl.append(ret_t)
        ret.append(ret_spl)
        
    return jnp.array(ret)


def test_time():
    jax.config.update("jax_enable_x64", True)
    key = random.PRNGKey(0)

    d_list = jnp.array([5, 10, 20,  200,  500])
    n_list = jnp.array([100, 200, 500,  1000, 20000])

    alp_list = jnp.array([.4, .5, .7, 1., 1.2])
    t_interval = jnp.array([.5, 1., 2., 5., 20.])
    

    # first test, fixed d = 100
    d_list = jnp.array([50])
    n_list = jnp.array([100, 200, 1000, 20000])

    alp_list = jnp.array([.5, 1.])

    all_ret_0 = {}
    for d in d_list:
        for n in n_list:
            print("Doing n=%d d=%d" % (n, d))
            if n <= d:
                continue
            for i_alp in range(alp_list.shape[0]):
                stf = Stiefel(n, d, alp_list[i_alp])
                ret = test_one_set(stf, key, t_interval, n_samples=5, n_repeats=2)
                all_ret_0[int(d), int(n), i_alp] = ret

    tbl = []
    for t_idx in range(t_interval.shape[0]):                
        for idx, val in all_ret_0.items():
            tbl.append([idx[1], idx[2], t_interval[t_idx]] + list(val[:, t_idx, :, :].mean(axis=((0, 1)))))

    raw_tbl = []
    for idx, val in all_ret_0.items():
        for t_idx in range(t_interval.shape[0]):
            for i_s in range(val.shape[0]):
                for i_r in range(val.shape[2]):
                    raw_tbl.append([idx[1], idx[2], t_interval[t_idx]] + list(val[i_s, t_idx, i_r, :]))
            
    pd.DataFrame(raw_tbl).to_pickle('by_n.pkl')
                    
    # second test, fix n = 2000, test d = [5, 10, 20, 200, 500]
    d_list = jnp.array([5, 10, 20, 200, 500])
    n_list = jnp.array([1000])

    alp_list = jnp.array([.5, 1.])

    all_ret_1 = {}
    for d in d_list:
        for n in n_list:
            print("Doing n=%d d=%d" % (n, d))
            if n <= d:
                continue
            for i_alp in range(alp_list.shape[0]):
                stf = Stiefel(n, d, alp_list[i_alp])
                ret = test_one_set(stf, key, t_interval, n_samples=5, n_repeats=2)
                all_ret_1[int(d), int(n), i_alp] = ret

    tbl1 = []
    for t_idx in range(t_interval.shape[0]):
        for idx, val in all_ret_1.items():
            tbl1.append([idx[0], idx[2], t_interval[t_idx]] + list(val[:, t_idx, :, :].mean(axis=((0, 1)))))

    raw_tbl1 = []
    for idx, val in all_ret_1.items():
        for t_idx in range(t_interval.shape[0]):
            for i_s in range(val.shape[0]):
                for i_r in range(val.shape[2]):
                    raw_tbl1.append([idx[0], idx[2], t_interval[t_idx]] + list(val[i_s, t_idx, i_r, :]))
            
    pd.DataFrame(raw_tbl1).to_pickle('by_d_1000.pkl')
            
                

    # third test, d=200, n=2000, alp_list = full append
    d_list = jnp.array([200])
    n_list = jnp.array([2000])
    alp_list = jnp.array([.4, .5, .7, 1., 1.2])
    
    all_ret_2 = {}
    for d in d_list:
        for n in n_list:
            if n <= d:
                continue
            for i_alp in range(alp_list.shape[0]):
                print("Doing alpha=%f " % alp_list[i_alp])
                
                stf = Stiefel(n, d, alp_list[i_alp])
                ret = test_one_set(stf, key, t_interval, n_samples=5, n_repeats=2)
                all_ret_2[int(d), int(n), i_alp] = ret

    tbl2 = []
    for t_idx in range(t_interval.shape[0]):
        for idx, val in all_ret_2.items():
            tbl2.append([idx[0], alp_list[idx[2]], t_interval[t_idx]] + list(val[:, t_idx, :, :].mean(axis=((0, 1)))))

    raw_tbl2 = []
    for idx, val in all_ret_1.items():
        for t_idx in range(t_interval.shape[0]):
            for i_s in range(val.shape[0]):
                for i_r in range(val.shape[2]):
                    raw_tbl2.append([idx[1], idx[2], t_interval[t_idx]] + list(val[i_s, t_idx, i_r, :]))
            
    pd.DataFrame(raw_tbl2).to_pickle('by_alp.pkl')

    
def display_test():
    import numpy as np
    jax.config.update("jax_enable_x64", True)   
    by_n_tbl = pd.read_pickle('by_n.pkl')
    # by_n_tbl.iloc[:, 2:] = np.array(by_n_tbl.iloc[:, 2:])
    by_n_tbl.iloc[:, 2] = [f"{a:04.1f}" for a in by_n_tbl.iloc[:, 2].values]
    by_n_tbl.columns = ['n', 'alp', 't', 'geo_time', 'par_time', 'err_geo', 'err_tan', 'err_eq']
    by_n_tbl['log_err_eq'] = [jnp.log10(a) for a in by_n_tbl.err_eq.values]

    by_n_prep = by_n_tbl.pivot_table(index=['alp', 'n'],
                                     columns='t',
                                     values=['par_time', 'log_err_eq'],
                                     aggfunc='mean')
    def str1(a):
        return '%.1f' % a    

    def str2(a):
        return '%.2f' % a    
    
    print(by_n_prep.to_latex(formatters=5*[str1] + 5*[str2]))
    # alp_tbl = jnp.array([.5, 1.])
    # by_n_tbl.loc[:, 'alp'] = alp_tbl[by_n_tbl.loc[:, 'alp'].values]
    by_d_tbl = pd.read_pickle('by_d.pkl')
    by_d_tbl.iloc[:, 2] = [f"{a:04.1f}" for a in by_d_tbl.iloc[:, 2].values]
    by_d_tbl.columns = ['d', 'alp', 't', 'geo_time', 'par_time', 'err_geo', 'err_tan', 'err_eq']

    by_d_tbl['log_err_eq'] = [jnp.log10(a) for a in by_d_tbl.err_eq.values]
    
    by_d_prep = by_d_tbl.pivot_table(index=['alp', 'd'],
                                     columns='t',
                                     values=['par_time', 'log_err_eq'],
                                     aggfunc='mean')

    print(by_d_prep.to_latex(formatters=5*[str1] + 5*[str2]))

    by_d_tbl = pd.read_pickle('by_d_1000.pkl')
    by_d_tbl.iloc[:, 2] = [f"{a:04.1f}" for a in by_d_tbl.iloc[:, 2].values]
    by_d_tbl.columns = ['d', 'alp', 't', 'geo_time', 'par_time', 'err_geo', 'err_tan', 'err_eq']

    by_d_tbl['log_err_eq'] = [jnp.log10(a) for a in by_d_tbl.err_eq.values]
    
    by_d_prep = by_d_tbl.pivot_table(index=['alp', 'd'],
                                     columns='t',
                                     values=['par_time', 'log_err_eq'],
                                     aggfunc='mean')

    print(by_d_prep.to_latex(formatters=5*[str1] + 5*[str2]))
    
            
                
def test_isometry():
    jax.config.update("jax_enable_x64", True)
    import jax.numpy.linalg as jla
    from jax_par_trans.expv.utils import (grand)
    
    key = random.PRNGKey(0)
    
    n = 1000
    d = 500
    alp = 1.
    stf = Stiefel(n, d, alp)
    x = jnp.zeros((n, d)).at[:d, :].set(jnp.eye(d))

    n_samples = 20
    tmp, key = grand(key, (n_samples, n_samples))
    q, _ = jla.qr(tmp)
    smpl = random.choice(key, n*d-(d**2+d)//2, (n_samples,), replace=False)

    all_smpl = []
    tridx = jnp.triu_indices(d, 1)
    for a in smpl:
        mat = jnp.zeros((n, d)).at[tridx[0][a], tridx[1][a]].set(1/jnp.sqrt(2*alp))
        if a < (d**2-d)//2:
            mat = mat.at[tridx[1][a], tridx[0][a]].set(-1/jnp.sqrt(2*alp))
        else:
            mat = jnp.concatenate(
                [jnp.zeros((d, d)),
                 jnp.zeros((n-d)*d).at[a - (d**2-d)//2].set(1.).reshape(n-d, d)],
                axis=0)
        all_smpl.append(mat)
    all_smpl = jnp.array(all_smpl)
    # sp1 = (q@all_smpl[:, None]).reshape(n_samples, n, d)
    all_smpl = (q@all_smpl.reshape(n_samples, -1)).reshape(n_samples, n, d)

    def cal_cov(gm, smpls):
        mat = jnp.zeros((n_samples,  n_samples))
        for i in range(n_samples):
            for j in range(i+1):
                mat = mat.at[i, j].set(stf.inner(gm, smpls[i, :, :], smpls[j, :, :]))
                if i != j:
                    mat = mat.at[j, i].set(mat[i, j])
        return mat
    
    print(cal_cov(x, all_smpl))

    v, key = stf.rand_vec(key, x)
    v = v/jnp.sqrt(stf.inner(x, v, v))
    transported = []
    t = 10.
    for i in range(n_samples):
        transported.append(stf.parallel(x, v, all_smpl[i, :, :], t))
        
    transported = jnp.array(transported)
    gm = stf.exp(x, t*v)
    cov = cal_cov(gm, transported)
    ei, ev  = jla.eigh(cov)
    print(ei)


def test_isometry():
    jax.config.update("jax_enable_x64", True)
    import matplotlib.pyplot as plt
    import jax.numpy.linalg as jla
    from jax_par_trans.expv.utils import (grand)
    
    key = random.PRNGKey(0)
    
    n = 1000
    d = 200
    alp = 1.
    stf = Stiefel(n, d, alp)
    x = jnp.zeros((n, d)).at[:d, :].set(jnp.eye(d))

    n_samples = 20

    all_smpl = []

    def normalize(a, x):
        return a / jnp.sqrt(stf.inner(x, a, a))
    
    for _ in range(n_samples):
        spl, key = stf.rand_vec(key, x)
        ft = random.choice(key, n_samples*3, (), replace=True)

        all_smpl.append(ft*normalize(spl, x))

    all_smpl = jnp.array(all_smpl)


    def cal_cov(gm, smpls):
        mat = jnp.zeros((n_samples,  n_samples))
        for i in range(n_samples):
            for j in range(i+1):
                mat = mat.at[i, j].set(stf.inner(gm, smpls[i, :, :], smpls[j, :, :]))
                if i != j:
                    mat = mat.at[j, i].set(mat[i, j])
        return mat
    
    cov_0 = cal_cov(x, all_smpl)

    v, key = stf.rand_vec(key, x)
    v = v/jnp.sqrt(stf.inner(x, v, v))

    cov_diff = []
    # t_grid = [0.5, 1., 2., 5., 10., 15.]
    t_grid = [0.1, .3, .5, .7, 1.2, 1.5, 1.7, 2.1, 3., 15.]
    for t in t_grid:
        transported = []
        for i in range(n_samples):
            transported.append(stf.parallel(x, v, all_smpl[i, :, :], t))

        transported = jnp.array(transported)
        gm = stf.exp(x, t*v)
        cov_t = cal_cov(gm, transported)
        # print(cz(cov_t- cov_0))
        cov_diff.append((t, cov_t))
    
    plt.plot(t_grid, [jnp.log10(cz(cov_diff[i][1] - cov_0)) for i in range(len(cov_diff))])
    plt.ylim(-14, 0)
    plt.xlabel("t(seconds)")
    plt.ylabel("max log10 of differences")    
    plt.title("Maximum absolute difference of inner product matrices")
    plt.savefig("cov_diff_stief.png")
    plt.show()


def test_stiefel_fast():
    # import jax
    # jax.config.update('jax_default_device', jax.devices('cpu')[0])
    from jax_par_trans.manifolds.stiefel import StiefelParallelOperator
    n = 1000
    d = 500

    key = random.PRNGKey(0)
    alp = .6
    stf = Stiefel(n, d, alp)
    x = jnp.concatenate([jnp.eye(d), jnp.zeros((n-d, d))])

    def normalize(x, ar):
        return ar/jnp.sqrt(stf.inner(x, ar, ar))
        
    ar, key = stf.rand_vec(key, x)
    b, key = stf.rand_vec(key, x)
    c, key = stf.rand_vec(key, x)

    ar = normalize(x, ar)
    b = normalize(x, b)
    c = normalize(x, c)

    salp = jnp.sqrt(alp)

    t = 1.2
    sp_opt = StiefelParallelOperator({"ar": ar, "alpha": alp})

    t0 = perf_counter()    
    w = stf._sc(sp_opt.expv(stf._sc(b, salp), t), 1/salp)
    t1 = perf_counter()
    # around 33 seconds
    print(1000*(t1 - t0))
