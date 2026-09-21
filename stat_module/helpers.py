import jax
import scipy

jax.config.update("jax_enable_x64", True)

import numpy           as np
import jax.numpy       as jnp

from scipy                 import stats
from scipy.optimize        import minimize
from cvxopt                import matrix, solvers
from numpyro.distributions import CategoricalProbs, MixtureSameFamily, Chi2

from stat_module import intermed

solvers.options['show_progress'] = False
solvers.options['maxiters'] = 200
solvers.options['feastol'] = 1e-7


def lower_right_inv(
        upp_left:  jnp.array,
        middle:    jnp.array,
        low_right: jnp.array,
    ) -> jnp.array:
    r"""Compute inverse of the lower right block of the given matrix.

    We partition our matrix into:

    ..math::
        mat = [[A, b], [b^t, C]]

    We are interested in computing the inverse of the lower right (q, q) block of mat (i.e. C^{-1}). Using the Schur complement of mat, we have:

    ..math::
        C^{-1} = (mat / A)^{-1} = (C - b^t A^{-1} b)^{-1}

    In most of our cases, this block is ill-conditioned, so we use a lower Cholesky decomposition (i.e. A = L L^t). We can then write:

    ..math::
        b^t A^{-1} b = b^t (L L^t)^{-1} b
                     = b^t (L^t)^{-1} L^{-1} b
                     = c^t c

    where we let :math:`c = L^{-1} b`. Note that :math:`L^{-1}b` is the solution to the system:

    ..math::
        Lx = b  \implies x = L^{-1} b

    so we can compute :math:`c = x` using the solvers in standard linear algebra libraries. We then have:

    .. math::
        C^{-1} = (D - c^t c)^{-1}

    ARGS:
        upp_left  - jax.numpy array; the upper left block of the information matrix corresponding to the fixed effects covariances
        middle    - jax.numpy array; the block of the information matrix corresponding to the fixed effects/variance components covariances
        low_right - jax.numpy array; the lower right block of the information matrix corresponding to the variance components covariances
    RETURNS:
        jax.numpy array; the lower right block of the inverse matrix (q, q)
    RAISES:
        None
    """
    mean_val = jnp.mean(upp_left)
    upp_block = upp_left / mean_val
    mid_scale = middle / jnp.sqrt(mean_val)
    
    def wrap_pinv(mat):
        return jnp.linalg.pinv(mat)
    
    #inv_func = jax.vmap(wrap_inv, in_axes = [0])
    pinv_func = jax.vmap(wrap_pinv, in_axes = [0])
    ## (rel_targs, p, p)
    upp_block_invs = pinv_func(upp_block)
    ## replace non-invertible entries with pseudo-inverses

    ## D - C A^{-1} B = M/A 
    def wrap_func(c_i, a_i):
        ## c_i is (p, q)
        ## a_i is (p, p)
        return jnp.matmul(jnp.matmul(c_i.T, a_i), c_i)
    vmap_func = jax.vmap(wrap_func, in_axes = [0, 0])
    ## (n_targs, q, q)
    vals = vmap_func(mid_scale, upp_block_invs)
    schur_comp = low_right - jnp.sum(vals, axis = 0)
    ## D^{-1} = (M/A)^{-1}
    return jnp.linalg.inv(schur_comp)





def compute_infimum(
        score_vec: jnp.array, 
        info_mat:  jnp.array,
        test_p:    int,
        q:         int
    ) -> float:
    """Compute the infimum of the quadratic program under the constraint that var_comp is non-negative...
    
    Uses `cvxopt` package's solvers.qp (quadratic program solver).

    ARGS:
        score_vec     - jax.numpy array; the gradient (first derivative) of the full marginal log-likelihood with respect to var_comp evaluated at var_comp = 0 (q,)
        info_mat - jax.numpy array; the inverse Schur complement of the lower right block of the Fisher information (q + test_p, q + test_p)
        test_p    - int; the number of fixed effects being tested
        q         - int; dimension of variance component (var_comp)
    RETURNS:
        inf_val - jnp.array; the value of the objective function evaluated at the solution found by the solver (q + test_p, 1)
    RAISES:
        None
    """
    ## invert information
    info_inv = jnp.linalg.pinv(info_mat)
    ## objective function
    dims = q
    q_arr = np.array(-jnp.linalg.solve(info_mat, score_vec), dtype = float)
    p_arr = np.array(info_inv, dtype = float)
    q_vec = matrix(q_arr)
    p_arr = matrix(p_arr)
    ## inequality constraint on var_comp
    # g_arr = matrix(-1.0 * np.eye(q))
    var_comp_const = [-1.0] * q
    fx_effs_const = [0.0] *0
    g_arr = matrix(np.diag(var_comp_const + fx_effs_const))
    ## not exactly zero to account for feasibility tolerance
    h_vec = matrix([1.0e-7] * dims) 
    ## solve the quadratic form
    sol = solvers.qp(p_arr, q_vec, g_arr, h_vec)
    sol_x = jnp.squeeze(jnp.array(sol['x']))
    ## score offset
    score_os = score_vec - sol_x
    inf_val = score_os.T @ info_inv @ score_os
    return inf_val







# def compute_infimum(
#         score:     jnp.array, 
#         schur_inv: jnp.array,
#         test_p:    int,
#         q:         int
#     ) -> float:
#     """Compute the infimum of the quadratic program under the constraint that var_comp is non-negative...
    
#     Uses `cvxopt` package's solvers.qp (quadratic program solver).

#     ARGS:
#         score     - jax.numpy array; the gradient (first derivative) of the full marginal log-likelihood with respect to var_comp evaluated at var_comp = 0 (q,)
#         schur_inv - jax.numpy array; the inverse Schur complement of the lower right block of the Fisher information (q + test_p, q + test_p)
#         test_p    - int; the number of fixed effects being tested
#         q         - int; dimension of variance component (var_comp)
#     RETURNS:
#         inf_val - jnp.array; the value of the objective function evaluated at the solution found by the solver (q + test_p, 1)
#     RAISES:
#         None
#     """
#     ## objective function
#     dims = test_p + q
#     if dims == 1 and not hasattr(score, 'shape'):
#         score = jnp.array([score])
#     q_arr = np.array(-jnp.matmul(schur_inv, score), dtype = float)
#     p_arr = np.array(schur_inv, dtype = float)
#     q_vec = matrix(q_arr)
#     p_arr = matrix(p_arr)
#     ## inequality constraint on var_comp
#     # g_arr = matrix(-1.0 * np.eye(q))
#     var_comp_const = [-1.0] * q
#     fx_effs_const = [0.0] * test_p
#     g_arr = matrix(np.diag(var_comp_const + fx_effs_const))
#     ## not exactly zero to account for feasibility tolerance
#     h_vec = matrix([1.0e-7] * dims) 
#     ## solve the quadratic form
#     sol = solvers.qp(p_arr, q_vec, g_arr, h_vec)
#     sol_x = jnp.squeeze(jnp.array(sol['x']))
#     ## compute infimum value
#     if dims == 1:
#         inf_val = jnp.matmul(
#             jnp.matmul((score - sol_x), schur_inv),
#             (score - sol_x)
#         )
#     else:
#         inf_val = jnp.matmul(
#             jnp.matmul(jnp.transpose((score - sol_x)), schur_inv),
#             (score - sol_x)
#         )
#     return inf_val


##----------------------------------------------------------------------------##
## Recurrence relations.                                                      ##
##----------------------------------------------------------------------------##
def digamma_diff(y_i, disp_i):
    """
    Digamma difference recurrence relation for target i values.
    
    digamma(y + disp) - digamma(disp) is equal to sum from k = 0 to y-1 of 1 / (disp + k)
    """
    y_i = y_i.astype('int')
    ks = jnp.arange(jnp.max(y_i))
    summands = 1 / (disp_i + ks)
    values = jnp.concatenate([jnp.zeros(1), jnp.cumsum(summands)])
    return values[y_i]

def trigamma_diff(y_i, disp_i):
    """
    Trigamma difference recurrence relation for target i values.
    
    trigamma(y + disp) - trigamma(disp) is equal to sum from k = 0 to y-1 of -1 / (disp + k)^2
    """
    y_i = y_i.astype('int')
    ks = jnp.arange(jnp.max(y_i))
    summands = -1 / (disp_i + ks)**2
    values = jnp.concatenate([jnp.zeros(1), jnp.cumsum(summands)])
    return values[y_i]


##----------------------------------------------------------------------------##
## Expectations (negative binomial PMF).                                      ##
##----------------------------------------------------------------------------##
def digamma_diff_exp(mu_i, disp_i, tol = 1e-12):
    r"""
    Approximate the following:
    
    E[digamma(y + disp) - digamma(disp)] = \sum_{r = 0}^\infty P(y > r) / (disp + r). 
    
    Truncates the sum to the cover 1 - tol probability mass AND so that summand contribution is < tol.

    ARGS:
        mu_i - jnp.array; estimated means for target i
        disp_i - jnp.array; estimate dispersion parameter for target i
        tol - float; approximation error tolerance
    RETURNS:
        jnp.array (n_samps,)
    """
    ## scipy.stats uses (n,p) parameterization
    n = disp_i
    p_vals = 1 / (1 + (1 / disp_i) * mu_i)
    # max_vals = scipy.stats.nbinom.isf(tol, n, p_vals)
    max_vals = scipy.stats.nbinom.ppf(1.0 - 1e-12, disp_i, disp_i / (disp_i + mu_i))
    ## check contribution is small enough for "good" approximation
    contrb =  max(1 / (disp_i + max_vals) * scipy.stats.nbinom.sf(max_vals, n, p_vals))
    max_vals = np.ceil(max_vals).astype(int) ## to int
    if contrb > tol:
        max_vals += 1
        contrb = max(1 / (disp_i + max_vals) * max(scipy.stats.nbinom.sf(max_vals, n, p_vals)))
    ## approximate expectation as truncated sum
    def exp_arg(max_val, p_val):
        r_vals = range(int(max_val))
        probs = scipy.stats.nbinom.sf(r_vals, n, p_val)
        vals = 1 / (disp_i + jnp.array(r_vals))
        return jnp.sum(vals * probs)
    
    exp_vals = jnp.array(
        [exp_arg(max_val_i, p_val_i) for max_val_i, p_val_i in zip(max_vals, p_vals)]
    )
    return exp_vals

def trigamma_diff_exp(mu_i, disp_i, tol = 1e-12):
    r"""
    Approximate the following:
    
    E[trigamma(y + disp) - trigamma(disp)] = \sum_{r = 0}^\infty P(y > r) / (disp + r)^2. 
    
    Truncates the sum to the cover 1 - tol probability mass AND so that summand contribution is < tol.

    ARGS:
        mu_i - jnp.array; estimated means for target i
        disp_i - jnp.array; estimate dispersion parameter for target i
        tol - float; approximation error tolerance
    RETURNS:
        jnp.array (n_samps,)
    """
    ## scipy.stats uses (n,p) parameterization
    n = disp_i
    p_vals = 1 / (1 + (1 / disp_i) * mu_i)
    #max_vals = scipy.stats.nbinom.isf(tol, n, p_vals)
    max_vals = scipy.stats.nbinom.ppf(1.0 - 1e-12, disp_i, p_vals)
    max_vals = np.ceil(max_vals).astype(int) ## to int
    ## check contribution is small enough for "good" approximation
    # contrb =  max(1 / (disp_i + max_vals) * scipy.stats.nbinom.sf(max_vals, n, p_vals))
    # if contrb > tol:
    #     max_vals += 1
    #     contrb = max(1 / (disp_i + max_vals) * max(scipy.stats.nbinom.sf(max_vals, n, p_vals)))
    ## approximate expectation as truncated sum
    def exp_arg(k, p_val):
        r_vals = jnp.arange(k)
        probs = scipy.stats.nbinom.sf(r_vals, n, p_val)
        vals = 1 / (disp_i + jnp.array(r_vals))**2
        return -jnp.sum(vals * probs)
    
    exp_vals = jnp.array(
        [exp_arg(max_val_i, p_val_i) for max_val_i, p_val_i in zip(max_vals, p_vals)]
    )
    return exp_vals



def digamma_y_ctr_exp(mu_i, disp_i, tol = 1e-12):
    r"""
    Approximate the following:
    
    E[digamma(y + disp)(y - mu)^2] = 
    
    Truncates the sum to the cover 1 - tol probability mass AND so that summand contribution is < tol.

    ARGS:
        mu_i - jnp.array; estimated means for target i
        disp_i - jnp.array; estimate dispersion parameter for target i
        tol - float; approximation error tolerance
    RETURNS:
        jnp.array (n_samps,)
    """
    ## scipy.stats uses (n,p) parameterization
    n = disp_i
    p_vals = 1 / (1 + (1 / disp_i) * mu_i)
    # max_vals = scipy.stats.nbinom.isf(tol, n, p_vals)
    max_vals = scipy.stats.nbinom.ppf(1.0 - 1e-12, disp_i, disp_i / (disp_i + mu_i))
    max_vals = np.ceil(max_vals).astype(int) ## to int
    # ## check contribution is small enough for "good" approximation
    # contrb =  max(1 / (disp_i + max_vals) * scipy.stats.nbinom.sf(max_vals, n, p_vals))
    # if contrb > tol:
    #     max_vals += 1
    #     contrb = max(1 / (disp_i + max_vals) * max(scipy.stats.nbinom.sf(max_vals, n, p_vals)))
    ## approximate expectation as truncated sum
    def exp_arg(k, mu_ij, p_val):
        r_vals = jnp.arange(k + 1)
        probs = scipy.stats.nbinom.pmf(r_vals, n, p_val)
        vals = (r_vals - mu_ij) * scipy.special.digamma(r_vals + disp_i)
        return jnp.sum(vals * probs)
    
    exp_vals = jnp.array(
        [exp_arg(max_val_i, mu_ij, p_val_i) for max_val_i, mu_ij, p_val_i in zip(max_vals, mu_i, p_vals)]
    )
    return exp_vals


def digamma_y_ctr_sqr_exp(mu_i, disp_i, tol = 1e-12):
    r"""
    Approximate the following:
    
    E[digamma(y + disp)(y - mu)^2] = 
    
    Truncates the sum to the cover 1 - tol probability mass AND so that summand contribution is < tol.

    ARGS:
        mu_i - jnp.array; estimated means for target i
        disp_i - jnp.array; estimate dispersion parameter for target i
        tol - float; approximation error tolerance
    RETURNS:
        jnp.array (n_samps,)
    """
    ## scipy.stats uses (n,p) parameterization
    n = disp_i
    p_vals = 1 / (1 + (1 / disp_i) * mu_i)
    # max_vals = scipy.stats.nbinom.isf(tol, n, p_vals)
    max_vals = scipy.stats.nbinom.ppf(1.0 - 1e-12, disp_i, disp_i / (disp_i + mu_i))
    max_vals = np.ceil(max_vals).astype(int) ## to int
    # ## check contribution is small enough for "good" approximation
    # contrb =  max(1 / (disp_i + max_vals) * scipy.stats.nbinom.sf(max_vals, n, p_vals))
    # if contrb > tol:
    #     max_vals += 1
    #     contrb = max(1 / (disp_i + max_vals) * max(scipy.stats.nbinom.sf(max_vals, n, p_vals)))
    ## approximate expectation as truncated sum
    def exp_arg(k, mu_ij, p_val):
        r_vals = jnp.arange(k)
        probs = scipy.stats.nbinom.pmf(r_vals, n, p_val)
        vals = (r_vals - mu_ij)**2 * scipy.special.digamma(r_vals + disp_i)
        return jnp.sum(vals * probs)
    
    exp_vals = jnp.array(
        [exp_arg(max_val_i, mu_ij, p_val_i) for max_val_i, mu_ij, p_val_i in zip(max_vals, mu_i, p_vals)]
    )
    return exp_vals



##----------------------------------------------------------------------------##
## Cutoffs computations.                                                      ##
##----------------------------------------------------------------------------##
def est_prelim_cutoff(mu_arr, const = 1e-4):
    """
    Compute data-dependent threshold for regime switch in terms of gamma parameterization.
    """
    max_mu = jnp.max(mu_arr)
    return max_mu / const


def est_cutoff(mu_arr, const = 1e-4):
    """
    Compute data-dependent threshold for regime switch. 
    """
    ## TODO
    return


def check_stabil(mat_inv, rtol = 1e-10, atol = 1e-14):
    """
    Check stability of matrix to be inverted (mat_inv) to avoid numerical issues when computing orthogonalization.
    """
    mat_det = jnp.linalg.det(mat_inv)
    if abs(mat_det) <= atol + rtol * max(1.0, abs(mat_det)):
        return False
    return True









#################
# def poi_exp(arg_func, mu_ij):
#     return scipy.stats.poisson.expect(func = arg_func, args = (mu_ij,))

# def nb_exp(arg_func, mu_ij, disp_i):
#     disp_i = disp_i[0]
#     p = disp_i / (disp_i + mu_ij)
#     r = disp_i
#     return scipy.stats.nbinom.expect(func = arg_func, args = (r, p))









# def log_lik_ratio(draws, mu_i, disp_i):
#     """
#     Compute the natural logarithm of the ratio of the negative binomial and Poisson likelihoods.

#     ARGS:
#         draws - jnp.array (n_samps, d); d Poisson draws from Poisson(mu_ij) 
#         mu_i  - jnp.array (n_samps, 1); means for target i
#         disp  - jnp.array (1, 1); dispersion parameter estimate for target i
#     RETURNS:
#         jnp.array (n_samps,); log of ratio of NB and Poi likelihoods for samples from target i
#     """
#     if disp_i == float('inf'):
#         ## skip computations because this is exactly Poisson
#         return jnp.zeros(draws.shape)
#     else:
#         term1 = log_gamma_diff(draws, disp_i)
#         term2 = -disp_i * jnp.log1p(mu_i / disp_i)
#         term3 = -draws * jnp.log(disp_i + mu_i)
#         return term1 + term2 + term3 + mu_i


# def imp_samp(arg_func, mu_i, disp_i, rng, m = 1e4, diagnos = True):
#     """
#     Estimate the expectation of arg_func via importance sampling. Works well for large disp. 

#     ARGS:
#         mu_i   - jnp.array (n_samps, 1); means for target i
#         disp_i - jnp.array (1,); dispersion estimate for target i
#         rng    - numpy random number generator
#         m      - int; number of importance samples to draw
#         diagnos - bool; whether to return diagnostic information
#     """
#     ## (n_samps, m)
#     m = int(m)
#     if len(mu_i.shape) == 1:
#         mu_i = jnp.expand_dims(mu_i, 1)
#     draws = rng.poisson(mu_i, (mu_i.shape[0], m))
#     ## (n_samps,)
#     lr = jnp.exp(log_lik_ratio(draws, mu_i, disp_i))
#     estim = jnp.mean(arg_func(draws, disp_i) * lr, axis = 1)
#     if not diagnos:
#         return estim
#     ess = lr.sum()**2 / sum(lr**2)
#     mean_lr = lr.mean()
#     return estim, {'ess': ess, 'ess_frac': ess / m, 'mean_lr': mean_lr}


# def recur_rel(draws, disp, arg_func):
#     """
#     Compute recurrence relation for gamma/polygamma functions. For a particular function f, letting h := arg_func, we have the relation:
    
#     f(y + disp) - f(disp) = sum from k = 0 to y - 1 of { h(disp, k) }

#     For f(x) = digamma(x), h(x) = 1 / x.
#     For f(x) = trigamma(x), h(x) = 1 / x^2.

#     ARGS:
#         draws - jnp.array (n_samps, m); random sample
#         disp  - jnp.array (1,); dispersion parameter estimate for target i
#         arg_func - function; recurrence relation function
#     """
#     ## maximum poisson draw
#     max_draw = int(jnp.max(draws))
#     ## if zero then entire sum is zero
#     if max_draw == 0:
#         return jnp.zeros(draws.shape) 
#     ## additions within sum
#     k_idxs = jnp.arange(max_draw)
#     func_evals = arg_func(disp + k_idxs)
#     cumul_sum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(func_evals)])
#     return cumul_sum[draws]

# def log_gamma_diff(draws, disp):
#     """
#     Compute log Gamma(y + disp) - log Gamma(disp) for single value of disp and a random sample of y ~ Poi. The log Gamma() function exhibits the recurrence relation implemented in recur_rel() with arg_func(x) = ln(x).
#     """
#     return recur_rel(draws, disp, lambda x: jnp.log(x))

# def digamma_diff(draws, disp):
#     """
#     Compute digamma(y + disp) - digamma(disp) for single value of disp and a random sample of y ~ Poi. The digamma() function exhibits the recurrence relation implemented in recur_rel() with arg_func(x) = 1 / x.
#     """
#     return recur_rel(draws, disp, lambda x: 1.0 / x)

# def trigamma_diff(draws, disp):
#     """
#     Compute trigamma(y + disp) - trigamma(disp) for single value of disp and a random sample of y ~ Poi. The trigamma() function exhibits the recurrence relation implemented in recur_rel() with arg_func(x) = -1 / x^2.
#     """
#     return recur_rel(draws, disp, lambda x: -1.0 / x**2)


# def digamma_shift(draws, disp):
#     """
#     Compute digamma(y + gamma) as:
#         digamma(y + gamma) - digamma(gamma) + digamma(disp)
#     so that we can reuse the code above.
#     """
#     return digamma_diff(draws, disp) + scipy.special.digamma(disp)

# def trigamma_shift(draws, disp):
#     """
#     Compute trigamma(y + gamma) as:
#         trigamma(y + gamma) - trigamma(gamma) + trigamma(disp)
#     so that we can reuse the code above.
#     """
#     return trigamma_diff(draws, disp) + scipy.special.polygamma(1, disp)


# def compute_exp_digamma(model, m = 1e4):
#     """
#     Compute or approximate E[digamma(y + disp)] for all observations.
#     """
#     disp_arr = model.disp_arr.flatten()
#     mu_arr = intermed.compute_delta(model, inv = False)

#     def exp_digam(mu_i, disp_i):
#         if disp_i > 5000:
#             return imp_samp(
#                 arg_func = lambda x: digamma_shift(x, disp_i),
#                 mu_i     = mu_i, 
#                 disp_i   = disp_i, 
#                 m        = m, 
#                 rng      = model.data.rng, 
#                 diagnos  = False
#             )
#         else:
#             p = disp_i / (disp_i + mu_i)
#             r = disp_i
#             def exp_arg(x):
#                 return scipy.special.polygamma(1, x + disp_i)
#             return [scipy.stats.nbinom.expect(func = exp_arg, args = (r, prob)) for prob in p]
#     return np.stack([exp_digam(a, b) for a, b in zip(mu_arr, disp_arr)], axis = 0)


# def compute_exp_trigamma(model, m = 1e4):
#     """
#     Compute or approximate E[trigamma(y + disp)] for all observations.
#     """
#     disp_arr = model.disp_arr.flatten()
#     mu_arr = intermed.compute_delta(model, inv = False)

#     def exp_trigam(mu_i, disp_i):
#         if disp_i > 5000:
#             return imp_samp(
#                 arg_func = trigamma_diff, 
#                 mu_i     = mu_i, 
#                 disp_i   = disp_i, 
#                 m        = m, 
#                 rng      = model.data.rng, 
#                 diagnos  = False
#             )
#         else:
#             p = disp_i / (disp_i + mu_i)
#             r = disp_i
#             def exp_arg(x):
#                 return scipy.special.polygamma(1, x + disp_i)
#             return [scipy.stats.nbinom.expect(func = exp_arg, args = (r, prob)) for prob in p]
        
#     return np.stack([exp_trigam(a, b) for a, b in zip(mu_arr, disp_arr)], axis = 0)


# def compute_exp_digamma_y_ctr(model, m = 1e4):
#     """
#     Compute or approximate E[digamma(y + disp)(y - mu)] for all observations.
#     """
#     disp_arr = model.disp_arr.flatten()
#     mu_arr = intermed.compute_delta(model, inv = False)

#     def exp_digam_ctr(mu_i, disp_i):
#         if disp_i > 5000:
#             return imp_samp(
#                 arg_func = lambda x: digamma_shift(x, disp_i) * (x - mu_i),
#                 mu_i     = mu_i, 
#                 disp_i   = disp_i, 
#                 m        = m, 
#                 rng      = model.data.rng, 
#                 diagnos  = False
#             )
#         else:
#             p = disp_i / (disp_i + mu_i)
#             r = disp_i
#             def exp_arg(x, mu):
#                 return scipy.special.polygamma(1, x + disp_i) * (x - mu)
            
#             [exp_digam_ctr(a, b) for a, b in zip(mu_arr, disp_arr)]
#             return [scipy.stats.nbinom.expect(func = exp_arg, args = (r, prob)) for prob in p]
        
#     return np.stack([exp_digam_ctr(a, b) for a, b in zip(mu_arr, disp_arr)], axis = 0)




# def excpt_digamma_y(mu, disp, rng, n_samps = 1e4):
#     """
#     Estimate E[digamma(y + disp)(y - mu)] with importance sampling. Approximation will be bad for small disp.
#     """
#     h = lambda x: digamma_shift(x, disp) * (x - mu)
#     return imp_samp(
#         arg_func = h,
#         mu       = mu,
#         disp     = disp,
#         rng      = rng,
#         n_samps  = n_samps,
#         diagnos  = False
#     )

# def excpt_digamma_y2(mu, disp, rng, n_samps = 1e4):
#     """
#     Estimate E[digamma(y + disp)(y - mu)^2] with importance sampling. Approximation will be bad for small disp.
#     """
#     h = lambda x: digamma_shift(x, disp) * (x - mu)**2
#     return imp_samp(
#         arg_func = h,
#         mu       = mu,
#         disp     = disp,
#         rng      = rng,
#         n_samps  = n_samps,
#         diagnos  = False
#     )


def compute_score_test_p_val(test_stat, one_sided = False):
    """Compute p-value for score test statistic when q = 1."""
    one_df_p  = 1 - stats.chi2.cdf(test_stat, 1)
    if one_sided:
        return 0.5 * one_df_p 
    else:
        return one_df_p

## SAMPLING/QUANTILE FUNCTIONS
def chi_sqr_01_mix_quantiles(x):
    if x < 0.5:
        return 0.0
    return scipy.stats.chi2.ppf((x - 0.5) / 0.5, df = 1)

def chi_sqr_m_quantiles(x, m):
    return scipy.stats.chi2.ppf(x, df = m)

def chi_bar_sqr_sample(q, num_samps):
    weights      = [None] * (q + 1)
    components   = [None] * (q + 1)
    weights = (1 / 2**q) * jnp.array([scipy.special.comb(q, i) for i in range(q + 1)])
    components = Chi2(df = jnp.array(range(q + 1)))
    mixing_dist  = CategoricalProbs(probs = jnp.array(weights))
    mixture_dist = MixtureSameFamily(mixing_dist, components)
    sample = mixture_dist.sample(jax.random.key(0), sample_shape = (num_samps,))
    return sample


