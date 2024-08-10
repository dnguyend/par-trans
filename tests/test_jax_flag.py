import jax
import jax.numpy as jnp
from jax import jvp, random
from jax_par_trans.expv.utils import (cz, sym)
from jax_par_trans.manifolds import Flag
from time import perf_counter
import timeit

    
def test_flag():
    jax.config.update("jax_enable_x64", True)

    dvec = jnp.array([5, 2, 3])
    alp = .5
    flg = Flag(dvec, alp)
    key = random.PRNGKey(0)

    x, key = flg.rand_point(key)
    v, key = flg.rand_vec(key, x)
    va, key = flg.rand_vec(key, x)

    t = .8

    print(jvp(lambda x: flg.inner(x, va, va), (x,), (v,))[1])
    print(2*flg.inner(x, va, flg.christoffel_gamma(x, v, va)))

    r1 = jvp(lambda x: flg.proj(x, va), (x,), (v,))[1] + flg.christoffel_gamma(x, v, va)
    print(sym(x.T@r1))

    r1 = flg.exp(x, t*v)

    print(cz(jvp(lambda t: flg.exp(x, t*v), (t,), (1.,))[1]
             - flg.dexp(x, v, t, ddexp=False)[1]))

    print(cz(jvp(lambda t: flg.dexp(x, v, t)[1], (t,), (1.,))[1]
             - flg.dexp(x, v, t, ddexp=True)[2]))

    gmms = flg.dexp(x, v, t, ddexp=True)
    print(cz(gmms[2] + flg.christoffel_gamma(gmms[0], gmms[1], gmms[1])))

    Delta = flg.parallel_canonical(x, v, va, t)
    print(cz(sym(gmms[0].T@Delta)))
    print(cz(flg.proj_m(gmms[0].T@Delta) - gmms[0].T@Delta))

    print(cz(jvp(lambda t: flg.parallel_canonical(x, v, va, t), (t,), (1.,))[1] \
             + flg.christoffel_gamma(gmms[0], gmms[1], Delta)))

    print(flg.inner(x, va, va))
    print(flg.inner(gmms[0], Delta, Delta))


def test_big():
    jax.config.update("jax_enable_x64", True)

    dvec = jnp.array([200, 40, 60, 1700])
    alp = .5
    flg = Flag(dvec, alp)
    key = random.PRNGKey(0)

    x, key = flg.rand_point(key)
    v, key = flg.rand_vec(key, x)
    va, key = flg.rand_vec(key, x)

    t = .8

    print(cz(jvp(lambda x: flg.inner(x, va, va), (x,), (v,))[1] -
             2*flg.inner(x, va, flg.christoffel_gamma(x, v, va))))

    r1 = jvp(lambda x: flg.proj(x, va), (x,), (v,))[1] + flg.christoffel_gamma(x, v, va)
    print(cz(sym(x.T@r1)))
    print(cz(flg.proj_m(x.T@r1) - x.T@r1))

    r1 = flg.exp(x, t*v)

    print(cz(jvp(lambda t: flg.exp(x, t*v), (t,), (1.,))[1]
             - flg.dexp(x, v, t, ddexp=False)[1]))

    print(cz(jvp(lambda t: flg.dexp(x, v, t)[1], (t,), (1.,))[1]
             - flg.dexp(x, v, t, ddexp=True)[2]))

    gmms = flg.dexp(x, v, t, ddexp=True)
    print(cz(gmms[2] + flg.christoffel_gamma(gmms[0], gmms[1], gmms[1])))

    Delta = flg.parallel_canonical(x, v, va, t)
    print(cz(sym(gmms[0].T@Delta)))
    print(cz(flg.proj_m(gmms[0].T@Delta) - gmms[0].T@Delta))

    print(cz(jvp(lambda t: flg.parallel_canonical(x, v, va, t), (t,), (1.,))[1] \
             + flg.christoffel_gamma(gmms[0], gmms[1], Delta)))


    # compare wit Stiefel
    from jax_par_trans.manifolds import Stiefel
    stf = Stiefel(flg.shape[0], flg.shape[1], .5)

    t0 = perf_counter()
    stf.dexp(x, v, t, ddexp=True)[2]
    t1 = perf_counter()
    print(t1-t0)

    t2 = perf_counter()
    flg.dexp(x, v, t, ddexp=True)[2]
    t3 = perf_counter()
    print(t3-t2)

    t4 = perf_counter()
    stf.parallel(x, v, va, t)
    t5 = perf_counter()
    print(t5-t4)

    t6 = perf_counter()
    flg.parallel_canonical(x, v, va, t)
    t7 = perf_counter()
    print(t7-t6)
        

def test_one_set(flg, key, t_interval, n_samples=10, n_repeats=5):
    ret = []
    d = flg.shape[1]
    for _ in range(n_samples):
        ret_spl = []
        x, key = flg.rand_point(key)
        v, key = flg.rand_vec(key, x)
        va, key = flg.rand_vec(key, x)
        # compile the git
        flg.dexp(x, v, 1.)
        par = flg.parallel_canonical(x, v, va, 1.)

        for t in t_interval:
            ret_t = []
            for _ in range(n_repeats):
                t0 = perf_counter()
                gmms = flg.dexp(x, v, t)
                t1 = perf_counter()
                t_gmms = t1 - t0
                
                t3 = perf_counter()
                par = flg.parallel_canonical(x, v, va, t)
                t4 = perf_counter()
                t_par = t4 - t3

                # check accuracy:
                geo_man = cz(gmms[0].T@gmms[0] - jnp.eye(d))
                par_tan = cz(sym(gmms[0].T@par))
                par_eq = cz(jvp(lambda t: flg.parallel_canonical(x, v, va, t), (t,), (1.,))[1] +
                            flg.christoffel_gamma(gmms[0], gmms[1], par))

                ret_t.append([t_gmms, t_par, geo_man, par_tan, par_eq])                

            ret_spl.append(ret_t)
        ret.append(ret_spl)        
    return jnp.array(ret)

    
def test_time():
    jax.config.update("jax_enable_x64", True)
    key = random.PRNGKey(0)

    # scale this part by d
    dparts = [5, 4, 3]
    dbase = sum(dparts)
    d_list = jnp.array([12, 24, 48,  192, 492])
    n_list = jnp.array([100, 200, 500,  1000, 20000])

    t_interval = jnp.array([.5, 1., 2., 5., 20.])
    

    # first test, fixed d = 48
    d_list = jnp.array([48])
    n_list = jnp.array([100, 200, 1000, 20000])

    alp = .5
    
    all_ret_0 = {}
    for d in d_list:
        for n in n_list:
            print("Doing n=%d d=%d" % (n, d))
            if n <= d:
                continue
            dvec_d = d//dbase*jnp.array(dparts)
            dvec = jnp.concatenate([dvec_d, jnp.array([n-dvec_d.sum()])])
            print(dvec)
            flg = Flag(dvec, alp)
            ret = test_one_set(flg, key, t_interval, n_samples=5, n_repeats=2)
            all_ret_0[int(d), int(n)] = ret

    tbl = []
    for t_idx in range(t_interval.shape[0]):
        for idx, val in all_ret_0.items():
            tbl.append([idx[1], t_interval[t_idx]] + list(val[:, t_idx, :, :].mean(axis=((0, 1)))))

    raw_tbl = []
    for idx, val in all_ret_0.items():
        for t_idx in range(t_interval.shape[0]):
            for i_s in range(val.shape[0]):
                for i_r in range(val.shape[2]):
                    raw_tbl.append([idx[1], t_interval[t_idx]] + list(val[i_s, t_idx, i_r, :]))

    import pandas as pd
    pd.DataFrame(raw_tbl).to_pickle('flg_by_n.pkl')
                    
    # second test, fix n = 2000, test d = [5, 10, 20, 200, 500]
    d_list = jnp.array([12, 48, 192, 492])
    n_list = jnp.array([1000])

    all_ret_1 = {}
    for d in d_list:
        for n in n_list:
            print("Doing n=%d d=%d" % (n, d))
            if n <= d:
                continue
            dvec_d = d//dbase*jnp.array(dparts)
            dvec = jnp.concatenate([dvec_d, jnp.array([n-dvec_d.sum()])])
            print(dvec)
            flg = Flag(dvec, alp)
            ret = test_one_set(flg, key, t_interval, n_samples=5, n_repeats=2)
            all_ret_1[int(d), int(n)] = ret

    tbl1 = []
    for t_idx in range(t_interval.shape[0]):
        for idx, val in all_ret_1.items():
            tbl1.append([idx[0], t_interval[t_idx]] + list(val[:, t_idx, :, :].mean(axis=((0, 1)))))

    raw_tbl1 = []
    for idx, val in all_ret_1.items():
        for t_idx in range(t_interval.shape[0]):
            for i_s in range(val.shape[0]):
                for i_r in range(val.shape[2]):
                    raw_tbl1.append([idx[0], t_interval[t_idx]] + list(val[i_s, t_idx, i_r, :]))
            
    pd.DataFrame(raw_tbl1).to_pickle('flg_by_d_1000.pkl')
    

def display_test():
    import pandas as pd
    jax.config.update("jax_enable_x64", True)   
    by_n_tbl = pd.read_pickle('flg_by_n.pkl')
    # by_n_tbl.iloc[:, 2:] = np.array(by_n_tbl.iloc[:, 2:])
    by_n_tbl.iloc[:, 1] = [f"{a:04.1f}" for a in by_n_tbl.iloc[:, 1].values]
    by_n_tbl.columns = ['n', 't', 'geo_time', 'par_time', 'err_geo', 'err_tan', 'err_eq']
    by_n_tbl['log_err_eq'] = [jnp.log10(a) for a in by_n_tbl.err_eq.values]

    by_n_prep = by_n_tbl.pivot_table(index='n',
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
    by_d_tbl = pd.read_pickle('flg_by_d_1000.pkl')
    by_d_tbl.iloc[:, 1] = [f"{a:04.1f}" for a in by_d_tbl.iloc[:, 1].values]
    by_d_tbl.columns = ['d', 't', 'geo_time', 'par_time', 'err_geo', 'err_tan', 'err_eq']

    by_d_tbl['log_err_eq'] = [jnp.log10(a) for a in by_d_tbl.err_eq.values]
    
    by_d_prep = by_d_tbl.pivot_table(index='d',
                                     columns='t',
                                     values=['par_time', 'log_err_eq'],
                                     aggfunc='mean')

    print(by_d_prep.to_latex(formatters=5*[str1] + 5*[str2]))
    
    


def test_isometry():
    jax.config.update("jax_enable_x64", True)

    dvec = jnp.array([30, 20, 50, 900])
    # dvec = jnp.array([3, 2, 5])
    alp = .5

    key = random.PRNGKey(0)

    alp = 1.
    flg = Flag(dvec, alp)
    n = flg.n
    d = flg.d
    x = jnp.zeros((n, d)).at[:d, :].set(jnp.eye(d))

    n_samples = 20

    all_smpl = []
    for _ in range(n_samples):
        spl, key = flg.rand_vec(key, x)
        all_smpl.append(spl)

    all_smpl = jnp.array(all_smpl)
    # sp1 = (q@all_smpl[:, None]).reshape(n_samples, n, d)

    def cal_cov(gm, smpls):
        mat = jnp.zeros((n_samples,  n_samples))
        for i in range(n_samples):
            for j in range(i+1):
                mat = mat.at[i, j].set(flg.inner(gm, smpls[i, :, :], smpls[j, :, :]))
                if i != j:
                    mat = mat.at[j, i].set(mat[i, j])
        return mat

    cov_0 = cal_cov(x, all_smpl)

    v, key = flg.rand_vec(key, x)
    v = v/jnp.sqrt(flg.inner(x, v, v))
    transported = []
    t = 10.
    for i in range(n_samples):
        transported.append(flg.parallel_canonical(x, v, all_smpl[i, :, :], t))

    transported = jnp.array(transported)
    gm = flg.exp(x, t*v)
    cov_t = cal_cov(gm, transported)
    print(cz(cov_t-cov_0))
    # ei, ev  = jla.eigh(cov)
    # print(ei)
    
