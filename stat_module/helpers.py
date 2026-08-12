import jax
import scipy

import numpy           as np
import jax.numpy       as jnp

from scipy                 import stats
from scipy.optimize        import minimize
from cvxopt                import matrix, solvers
from numpyro.distributions import CategoricalProbs, MixtureSameFamily, Chi2

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

    # ## if upper left shape is (_, 1, 1) and non-invertible, do pseudo-inverse
    # if upp_left.shape[-1] == 1:
    #     big_mat = jnp.diag(upp_block[:, 0, 0])
    #     if np.linalg.det(big_mat) == 0.0:
    #         upp_block_invs = np.linalg.pinv(big_mat)
    #     else:
    #         upp_block_invs = np.linalg.inv(big_mat)
    #     ## reshape to be (rel_targs, p, p)
    #     upp_block_invs = jnp.diag(upp_block_invs)
    #     upp_block_invs = jnp.expand_dims(upp_block_invs, axis = (1, 2))

    ## invert each matrix in block diagonal upp_left block --- this is kind of inefficient
    dets = jnp.linalg.det(upp_block)
    #zero_dets = np.where(dets == 0.0)

    def wrap_inv(mat):
        return jnp.linalg.inv(mat)
    
    def wrap_pinv(mat):
        return jnp.linalg.pinv(mat)
    
    #inv_func = jax.vmap(wrap_inv, in_axes = [0])
    pinv_func = jax.vmap(wrap_pinv, in_axes = [0])
    ## (rel_targs, p, p)
    upp_block_invs = pinv_func(upp_block)
    ## replace non-invertible entries with pseudo-inverses
    #upp_block_invs = upp_block_invs.at[zero_dets].set(pinv_func(upp_block[zero_dets]))

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

    # ## dimension of mat
    # dim = mat.shape[0]
    # ## get upper left block of matrix --- scale by mean for stability
    # #upp_block = mat[0:(dim - q), 0:(dim - q)]
    # upp_block = mat[:-q, :-q]
    # #mean = scipy.stats.gmean(jnp.ravel(upp_block))
    # mean = jnp.mean(upp_block)
    # upp_block_scale = upp_block / mean
    # ## rightmost chunk of matrix --- scale by sqrt mean to retain value
    # #r_col = mat[0:(dim - q), (dim - q):]
    # r_col = mat[:-q, -q:]
    # r_col_scale = r_col / jnp.sqrt(mean)
    # ## compute Cholesky factor
    # l_arr = jax.scipy.linalg.cho_factor(upp_block)
    # l_arr_scale = jax.scipy.linalg.cho_factor(upp_block_scale)
    # ## compute c vector as in description --- Cholesky solve
    # c_vec_scale = jax.scipy.linalg.cho_solve(l_arr_scale, r_col_scale)
    # ct_c_scale = jnp.matmul(jnp.transpose(c_vec_scale), c_vec_scale)
    # ## compute schur complement and invert
    # #low_block = mat[(dim - q):, (dim - q):]
    # low_block = mat[-q:, -q:]
    # schur_comp = low_block - ct_c_scale
    # return jnp.linalg.inv(schur_comp)


# def compute_infimum_scalar(
#         score:     jnp.array, 
#         schur_inv: jnp.array
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
#     def obj_func(b):
#         return (score - b)**2 * schur_inv

#     def jac_func(b):
#         return -2 * (score - b) * schur_inv
    
#     def ineq_constr(b):
#         return b

#     constraints = {
#         'type': 'ineq',
#         'fun':  ineq_constr,
#     }

#     inf_val = minimize(
#         fun         = obj_func, 
#         method      = 'COBYQA',
#         x0          = 1.0, 
#         constraints = constraints
#     )
#     print(inf_val)

#     return inf_val


def compute_infimum(
        score:     jnp.array, 
        schur_inv: jnp.array,
        test_p:    int,
        q:         int
    ) -> float:
    """Compute the infimum of the quadratic program under the constraint that var_comp is non-negative...
    
    Uses `cvxopt` package's solvers.qp (quadratic program solver).

    ARGS:
        score     - jax.numpy array; the gradient (first derivative) of the full marginal log-likelihood with respect to var_comp evaluated at var_comp = 0 (q,)
        schur_inv - jax.numpy array; the inverse Schur complement of the lower right block of the Fisher information (q + test_p, q + test_p)
        test_p    - int; the number of fixed effects being tested
        q         - int; dimension of variance component (var_comp)
    RETURNS:
        inf_val - jnp.array; the value of the objective function evaluated at the solution found by the solver (q + test_p, 1)
    RAISES:
        None
    """
    ## objective function
    dims = test_p + q
    if dims == 1 and not hasattr(score, 'shape'):
        score = jnp.array([score])
    q_arr = np.array(-jnp.matmul(schur_inv, score), dtype = float)
    p_arr = np.array(schur_inv, dtype = float)
    q_vec = matrix(q_arr)
    p_arr = matrix(p_arr)
    ## inequality constraint on var_comp
    # g_arr = matrix(-1.0 * np.eye(q))
    var_comp_const = [-1.0] * q
    fx_effs_const = [0.0] * test_p
    g_arr = matrix(np.diag(var_comp_const + fx_effs_const))
    ## not exactly zero to account for feasibility tolerance
    h_vec = matrix([1.0e-7] * dims) 
    ## solve the quadratic form
    sol = solvers.qp(p_arr, q_vec, g_arr, h_vec)
    sol_x = jnp.squeeze(jnp.array(sol['x']))
    ## compute infimum value
    if dims == 1:
        inf_val = jnp.matmul(
            jnp.matmul((score - sol_x), schur_inv),
            (score - sol_x)
        )
    else:
        inf_val = jnp.matmul(
            jnp.matmul(jnp.transpose((score - sol_x)), schur_inv),
            (score - sol_x)
        )
    return inf_val


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


