import jax
import scipy

import numpy           as np
import jax.numpy       as jnp
from score_module import score, info

from scipy                 import stats
from cvxopt                import matrix, solvers
from numpyro.distributions import CategoricalProbs, MixtureSameFamily, Chi2

solvers.options['show_progress'] = False
solvers.options['maxiters'] = 200
solvers.options['feastol'] = 1e-7

def lower_right_inv(
        mat: jnp.array,
        q:   int  = 1
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
        mat:
            jax.numpy array; the matrix of interest
        q:
            int; dimension to consider as the "lower right block"
    RETURNS:
        jax.numpy array; the lower right block of the inverse matrix (q, q)
    RAISES:
        None
    """
    ## dimension of mat
    dim = mat.shape[0]
    ## get upper left block of matrix --- scale by mean for stability
    #upp_block = mat[0:(dim - q), 0:(dim - q)]
    upp_block = mat[:-q, :-q]
    #mean = scipy.stats.gmean(jnp.ravel(upp_block))
    mean = jnp.mean(upp_block)
    upp_block_scale = upp_block / mean
    ## rightmost chunk of matrix --- scale by sqrt mean to retain value
    #r_col = mat[0:(dim - q), (dim - q):]
    r_col = mat[:-q, -q:]
    r_col_scale = r_col / jnp.sqrt(mean)
    ## compute Cholesky factor
    l_arr = jax.scipy.linalg.cho_factor(upp_block)
    l_arr_scale = jax.scipy.linalg.cho_factor(upp_block_scale)
    ## compute c vector as in description --- Cholesky solve
    c_vec_scale = jax.scipy.linalg.cho_solve(l_arr_scale, r_col_scale)
    ct_c_scale = jnp.matmul(jnp.transpose(c_vec_scale), c_vec_scale)
    ## compute schur complement and invert
    #low_block = mat[(dim - q):, (dim - q):]
    low_block = mat[-q:, -q:]
    schur_comp = low_block - ct_c_scale
    return jnp.linalg.inv(schur_comp)


def compute_infimum(
        score:     jnp.array, 
        schur_inv: jnp.array,
        test_p:    int,
        q:         int
    ) -> float:
    """Compute the infimum of the quadratic program under the constraint that tau_sqr is non-negative...
    
    Uses `cvxopt` package's solvers.qp (quadratic program solver).

    ARGS:
        score     - jax.numpy array; the gradient (first derivative) of the full marginal log-likelihood with respect to tau_sqr evaluated at tau_sqr = 0 (q,)
        schur_inv - jax.numpy array; the inverse Schur complement of the lower right block of the Fisher information (q + test_p, q + test_p)
        test_p    - int; the number of fixed effects being tested
        q         - int; dimension of variance component (tau_sqr)
    RETURNS:
        inf_val - jnp.array; the value of the objective function evaluated at the solution found by the solver (q + test_p, 1)
    RAISES:
        None
    """
    ## objective function
    dims = test_p + q
    if dims == 1:
        score = jnp.array([score])
    q_arr = np.array(-jnp.matmul(schur_inv, score), dtype = float)
    p_arr = np.array(schur_inv, dtype = float)
    q_vec = matrix(q_arr)
    p_arr = matrix(p_arr)
    ## inequality constraint on tau_sqr
    # g_arr = matrix(-1.0 * np.eye(q))
    tau_sqr_const = [-1.0] * q
    fx_effs_const = [0.0] * test_p
    g_arr = matrix(np.diag(tau_sqr_const + fx_effs_const))
    ## not exactly zero to account for feasibility tolerance
    h_vec = matrix([1.0e-7] * dims) 
    ## solve the quadratic form
    sol = solvers.qp(p_arr, q_vec, g_arr, h_vec)
    #print(p_arr, q_vec, g_arr, h_vec)
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
def chi_sqr_01_mix_sample(n):
    rng   = np.random.default_rng(None)
    chi_sqr = rng.chisquare(df = 1, size = n)
    coin_flip = rng.binomial(n = 1, p = 0.5, size = n)
    return chi_sqr * coin_flip

def chi_sqr_01_mix_quantiles(x):
    if x < 0.5:
        return 0.0
    return scipy.stats.chi2.ppf((x - 0.5) / 0.5, df = 1)

def chi_bar_12_mix_quantiles(num_samps):
    weights      = [0.5, 0.5]
    components   = [None] * 2
    components = Chi2(df = jnp.array(range(1, 3)))
    mixing_dist  = CategoricalProbs(probs = jnp.array(weights))
    mixture_dist = MixtureSameFamily(mixing_dist, components)
    sample = mixture_dist.sample(jax.random.key(0), sample_shape = (num_samps,))
    return sample



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


