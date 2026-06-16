"""Computations related to the score function."""

import math
import jax
import jax.numpy    as jnp

from stat_module import info, helpers, intermed

from routine_module import doc

def compute_score_var_comp(
        model,
        scale: bool = True
    ):
    r"""Compute the first derivative of the (full) marginal log-likelihood with respect to variance component vector evaluated at the provided parameter values. Equation 9 in Lin (1997).

    ARGS:
        model - instance of Model class
        scale - boolean; whether to scale by :math:`\frac{1}{\sqrt{k}}`. default: True
    RETURNS:
        jax.numpy array; the first derivative of the full data's marginal log-likelihood with respect to variance component vector (q,)
        float; scaling value (1 if scale == False)
    RAISES:
        NONE
    """
    scores = jnp.zeros((model.data.q))

    ## initial computations
    resid_vals     = model.compute_resid()
    delta_inv_omega_vals = model.compute_delta_inv_omega()
    xi_vals = model.compute_xi()

    for idx in range(model.data.q):
        ## (n_samps, n_samps) --- same for all targets
        a_j_arr = intermed.compute_a_j(model, j = idx)

        ## vectorize wrapper --- compute target-by-target
        def wrap_func(resid_targ, delta_invomega_targ, xi_targ):
            rdo_targ = resid_targ * delta_inv_omega_targ
            term1 = rdo_targ.T @ a_j_arr @ rdo_targ
            term2 = jnp.diag(xi_targ) @ a_j_arr
            return term1 - jnp.trace(term2)
    
        ## compute values
        vmap_func = jax.vmap(wrap_func, in_axes = [0, 0, 0])
        summands = vmap_func(resid_vals, delta_inv_omega_vals, xi_vals)
        full_log_lik = (1 / 2) * jnp.sum(summands)
        scores = scores.at[idx].set(full_log_lik)
    if scale:
        ## 2-norm
        scale_val = jnp.linalg.norm(scores)
    else:
        scale_val = 1
    wrt_var_comp = jnp.squeeze(scores / scale_val)
    
    if math.isnan(wrt_var_comp):
        doc.debug_sample(model)
        raise Exception
    return wrt_var_comp, scale_val




# def compute_score_var_comp(
#         model,
#         scale: bool = True
#     ):
#     r"""Compute the first derivative of the (full) marginal log-likelihood with respect to variance component vector evaluated at the provided parameter values. Equation 9 in Lin (1997).

#     ARGS:
#         model - instance of Model class
#         scale - boolean; whether to scale by :math:`\frac{1}{\sqrt{k}}`. default: True
#     RETURNS:
#         jax.numpy array; the first derivative of the full data's marginal log-likelihood with respect to variance component vector (q,)
#         float; scaling value (1 if scale == False)
#     RAISES:
#         NONE
#     """
#     scores = jnp.zeros((model.data.q))

#     ## initial computations
#     resid_vals     = model.compute_resid()
#     omega_vals     = intermed.compute_omega(model)
#     delta_inv_vals = intermed.compute_delta(model, inv = True)
#     xi_vals        = intermed.compute_xi(model)

#     for idx in range(model.data.q):
#         ## (n_samps, n_samps) --- same for all targets
#         a_j_arr = intermed.compute_a_j(model, j = idx)

#         ## vectorize wrapper --- compute target-by-target
#         def wrap_func(resid_targ, delta_inv_targ, omega_targ, xi_targ):
#             rdo_targ = resid_targ * delta_inv_targ * omega_targ
#             term1 = rdo_targ.T @ a_j_arr @ rdo_targ
#             term2 = jnp.diag(xi_targ) @ a_j_arr
#             return term1 - jnp.trace(term2)
    
#         ## compute values
#         vmap_func = jax.vmap(wrap_func, in_axes = [0, 0, 0, 0])
#         summands = vmap_func(resid_vals, delta_inv_vals, omega_vals, xi_vals)
#         full_log_lik = (1 / 2) * jnp.sum(summands)
#         scores = scores.at[idx].set(full_log_lik)
#     if scale:
#         ## 2-norm
#         scale_val = jnp.linalg.norm(scores)
#     else:
#         scale_val = 1
#     wrt_var_comp = jnp.squeeze(scores / scale_val)
    
#     if math.isnan(wrt_var_comp):
#         doc.debug_sample(model)
#         raise Exception
#     return wrt_var_comp, scale_val



# def compute_score(
#         model,
#         scale: bool = True
#     ):
#     r"""
#     Compute the first derivative of the (full) marginal log-likelihood with respect to the fixed effects and variance components evaluated at the parameter MLEs/values under H0.

#     ARGS:
#         model  - instance of Model class
#         scale         - boolean; whether to scale by :math:`\frac{1}{\sqrt{k}}`. default: True
#     RETURNS:
#         jax.numpy array; the first derivative of the full data's marginal log-likelihood with respect to fixed effects and variance component(s) (p* + q,)
#         float; scaling value (1 if scale == False)
#     RAISES:
#         NONE
#     """
#     var_comp, _ = compute_score_var_comp(
#         model,
#         scale = False
#     )
#     fx_effs, _  = compute_score_fx_effs(
#         model,
#         scale = False
#     )
#     ## combine
#     full_score = jnp.concatenate([fx_effs, jnp.reshape(var_comp, (model.data.q,))])
#     if scale:
#         ## 2-norm
#         scale_val = jnp.linalg.norm(full_score)
#     else:
#         scale_val = 1
#     scores = full_score / scale_val
#     return scores, scale_val
