"""Computations related to the score function."""

import math
import jax
import jax.numpy    as jnp

from score_module import info, intermed, helpers



def compute_score_var_comp(
        model,
        scale: bool = True
    ):
    r"""Compute the first derivative of the (full) marginal log-likelihood with respect to variance component vector evaluated at the provided parameter values. Equation 9 in Lin (1997).

    ARGS:
        model         - instance of Model class
        fx_effs_arr   - jnp.array; matrix of fixed effects estimates (n, k, p)
        tau_sqr       - jnp.array; fixed intercept matrix (1,)
        scale         - boolean; whether to scale by :math:`\frac{1}{\sqrt{k}}`. default: True
    RETURNS:
        jax.numpy array; the first derivative of the full data's marginal log-likelihood with respect to variance component vector (q,)
        float; scaling value (1 if scale == False)
    RAISES:
        NONE
    """
    scores = jnp.zeros((model.data.q))

    ## initial computations
    resid_vals     = model.compute_resid()
    omega_vals     = intermed.compute_omega(model)
    delta_inv_vals = intermed.compute_delta(model, inv = True)
    xi_vals        = intermed.compute_xi(model)

    for idx in range(model.data.q):
        ## (n_samps, n_samps) --- same for all targets
        a_j_arr = intermed.compute_a_j(model, j = idx)

        ## vectorize wrapper --- compute target-by-target
        def wrap_func(resid_targ, delta_inv_targ, omega_targ, xi_targ):
            rdo_targ = resid_targ * delta_inv_targ * omega_targ
            term1 = rdo_targ.T @ a_j_arr @ rdo_targ
            term2 = jnp.diag(xi_targ) @ a_j_arr
            return term1 - jnp.trace(term2)
    
        ## compute values
        vmap_func = jax.vmap(wrap_func, in_axes = [0, 0, 0, 0])
        summands = vmap_func(resid_vals, delta_inv_vals, omega_vals, xi_vals)
        full_log_lik = (1 / 2) * jnp.sum(summands)
        scores = scores.at[idx].set(full_log_lik)
    if scale:
        ## 2-norm
        scale_val = jnp.linalg.norm(scores)
    else:
        scale_val = 1
    wrt_tau_sqr = jnp.squeeze(scores / scale_val)
    
    if math.isnan(wrt_tau_sqr):
        raise Exception
    return wrt_tau_sqr, scale_val



def compute_test_stat(model, one_sided = False):
    """
    Compute score test statistic.
    
    ARGS:
        model - instance of Model class
        one_sided - boolean flag; whether to compute the one-sided score test
    RETURNS:
        float; the test statistic value
    RAISES:
        None
    """
    ## compute score
    score_val, scale_val = compute_score_var_comp(
        model, 
        scale = model.scale
    )
    ## compute information
    info_val = info.compute_info(
        model, 
        scale_val = scale_val
    )
    ## schur complement
    schur_inv = helpers.lower_right_inv(
        mat = info_val,
        q   = model.data.q
    )

    ## find infimum if one-sided test. otherwise set to 0
    inf_val = 0
    if one_sided:
        try:
            inf_val = helpers.compute_infimum(
                score     = score_val,
                schur_inv = schur_inv,
                test_p    = 0,
                q         = model.data.q
            )

        except:
            raise Exception("didn't work :(")
    
    ## compute stat
    if model.data.q == 1:
        stat = score_val**2 * schur_inv
    else:
        stat = jnp.matmul(
            jnp.matmul(jnp.transpose(score_val), schur_inv), score_val
        )
    return(jnp.squeeze(stat - inf_val))



def compute_score(
        model,
        scale: bool = True
    ):
    r"""
    Compute the first derivative of the (full) marginal log-likelihood with respect to the fixed effects and variance components evaluated at the provided parameter values. We only can compute this under H0 (i.e. tau_sqr = 0). 

    ARGS:
        model         - instance of Model class
        fx_effs_arr   - jnp.array; matrix of fixed effects estimates (n, k, p)
        scale         - boolean; whether to scale by :math:`\frac{1}{\sqrt{k}}`. default: True
    RETURNS:
        jax.numpy array; the first derivative of the full data's marginal log-likelihood with respect to fixed effects and variance component(s) (p* + q,)
        float; scaling value (1 if scale == False)
    RAISES:
        NONE
    """
    var_comp, _ = compute_score_var_comp(
        model,
        scale = False
    )
    fx_effs, _  = compute_score_fx_effs(
        model,
        scale = False
    )
    ## combine
    full_score = jnp.concatenate([fx_effs, jnp.reshape(var_comp, (model.data.q,))])
    if scale:
        ## 2-norm
        scale_val = jnp.linalg.norm(full_score)
    else:
        scale_val = 1
    scores = full_score / scale_val
    return scores, scale_val
