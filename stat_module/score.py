"""Computations related to the score function."""

import math
import jax
import jax.numpy    as jnp
import numpy        as np
from stat_module import info, helpers, intermed

from routine_module import doc

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
#     q = model.data.z_arr.shape[-1]
#     scores = jnp.zeros((q))

#     ## initial computations
#     resid_vals     = model.compute_resid()
#     xi_vals        = intermed.compute_xi(model)
#     delta_inv_omega_vals = intermed.compute_delta_inv_omega(model)
#     for idx in range(q):
#         ## (n_samps, n_samps) --- same for all targets
#         a_j_arr = intermed.compute_a_j(model, j = idx)

#         ## vectorize wrapper --- compute target-by-target
#         def wrap_func(resid_targ, delta_inv_omega_targ, xi_targ):
#             rdo_targ = resid_targ * delta_inv_omega_targ
#             term1 = rdo_targ.T @ a_j_arr @ rdo_targ
#             term2 = jnp.diag(xi_targ) @ a_j_arr
#             return term1 - jnp.trace(term2)
    
#         ## compute values
#         vmap_func = jax.vmap(wrap_func, in_axes = [0, 0, 0])
#         summands = vmap_func(resid_vals, delta_inv_omega_vals, xi_vals)
#         full_log_lik = (1 / 2) * jnp.sum(summands)
#         scores = scores.at[idx].set(full_log_lik)
#     if scale:
#         ## 2-norm
#         scale_val = jnp.linalg.norm(scores)
#     else:
#         scale_val = 1
#     wrt_var_comp = scores / scale_val

#     return wrt_var_comp, scale_val

def compute_score_var_comp(model) -> jnp.array:
    r"""Compute the first derivative of the (full) quasi-log-likelihood with respect to the variance components vector.

    ARGS:
        model - instance of Model class
    RETURNS:
        jax.numpy array (n_targs, q)
    RAISES:
        NONE
    """
    upsilon_vals = intermed.compute_delta_inv_omega(model) * model.compute_resid()
    xi_vals = intermed.compute_xi(model)
    z_arr = model.data.z_arr
    return (1 / 2) * (np.sum(upsilon_vals[:, :, None] * z_arr[None, :, :], axis = 1)**2 - np.sum(xi_vals[:, :, None] * z_arr[None, :, :]**2, axis = 1))


def compute_score_fx_effs(model) -> jnp.array:
    r"""Compute the first derivative of the (full) TRUE (negative binomial) log-likelihood with respect to the fixed effects parameter vector. 

    ARGS:
        model - instance of Model class
    RETURNS:
        jax.numpy array (n_targs, p)
    RAISES:
        NONE
    """
    upsilon_vals = intermed.compute_delta_inv_omega(model) * model.compute_resid()
    return upsilon_vals @ model.data.x_arr

def compute_score_disp(model) -> jnp.array:
    r"""Compute the first derivative of the (full) TRUE (negative binomial) log-likelihood with respect to the dispersion parameter vector. 

    ARGS:
        model - instance of Model class
    RETURNS:
        jax.numpy array (n_targs, 1)
    RAISES:
        NONE
    """
    ## digamma difference
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

    term1 = jnp.array([
        digamma_diff(y_i, disp_i) for y_i, disp_i in zip(model.data.construct_y(), model.disp_arr)
    ])
    term2 = jnp.log1p(model.compute_mu() / model.disp_arr)
    term3 = model.compute_resid() / (model.disp_arr + model.compute_mu())
    return jnp.sum(term1 - term2 - term3, axis = -1)[:, None]


def compute_score_eff(model, scale: bool = True) -> jnp.array:
    r"""
    Compute the efficient score vector for the variance components.

    ARGS:
        model - instance of Model class
    RETURNS:
        jax.numpy array (q, 1)
    RAISES:
        NONE
    """
    ## initial computations
    score_vc = compute_score_var_comp(model)
    score_f = compute_score_fx_effs(model)
    score_d = compute_score_disp(model)
    info_f_vc = info.info_fx_effs_var_comp(model)
    info_d_vc = info.info_disp_var_comp(model)
    info_ff = info.info_fx_effs_fx_effs(model)
    info_dd = info.info_disp_disp(model)
    ## corrections
    try:
        fx_crct = info_f_vc @ np.linalg.solve(info_ff, score_f[:, :, None])
    except:
        fx_crct = info_f_vc @ np.linalg.pinv(info_ff) @ score_f[:, :, None]
    ## only correct the ones with finite dispersion parameters
    mask = ~np.isinf(model.disp_arr).squeeze()
    d_crct = np.zeros((model.data.n_targs, model.data.z_arr.shape[-1], 1))
    d_crct_vals = info_d_vc[mask] @ np.linalg.solve(info_dd[mask][:, None, None], score_d[mask][:, None])
    d_crct[mask] = d_crct_vals
    # d_crct = info_d_vc @ np.linalg.solve(info_dd[:, None, None], score_d[:, :, None])
    scores = (score_vc[:, :, None] - fx_crct - d_crct).sum(axis = 0)
    if scale:
        ## 2-norm
        scale_val = jnp.linalg.norm(scores)
    else:
        scale_val = 1
    wrt_var_comp = scores / scale_val
    return wrt_var_comp, scale_val