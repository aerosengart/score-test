"""Computations related to the score function."""

import math
import jax
import scipy 

jax.config.update("jax_enable_x64", True)

import jax.numpy    as jnp
import numpy        as np

from stat_module import info, helpers, intermed

from routine_module import doc



##----------------------------------------------------------------------------##
## Score vector for fixed effects parameter.                                  ##
##----------------------------------------------------------------------------##

def score_fx_effs(model) -> jnp.array:
    r"""
    Compute the first derivative of the (true!) marginal log-likelihood with respect to fixed effects. Works regardless of magnitude of dispersion parameter.

    ARGS:
        model - instance of Model class
    RETURNS:
        jax.numpy array; (n_targs, p)
    """
    omega_delta_vals = intermed.compute_delta_inv_omega(model)
    resid_vals = model.compute_resid()
    upsilon_vals = omega_delta_vals * resid_vals
    return upsilon_vals @ model.data.x_arr


##----------------------------------------------------------------------------##
## Score vector for dispersion parameter.                                     ##
##----------------------------------------------------------------------------##
def score_disp_i_nb(y_i, mu_i, disp_i) -> jnp.array:
    r"""Compute the first derivative of the (true!) marginal log-likelihood with respect to the dispersion parameters (gamma) when the variance is mu + mu^2 / gamma.

    ARGS:
        model - instance of Model class
    RETURNS:
        jax.numpy array; (n_targs,)
    """
    dg_term = helpers.digamma_diff(y_i, disp_i)
    log_1p_term = jnp.log1p(mu_i / disp_i)
    resid_term = (y_i - mu_i) / (disp_i + mu_i)
    return jnp.sum(dg_term - log_1p_term - resid_term)

def score_disp_i_poi(y_i, mu_i, disp_i) -> jnp.array:
    r"""Compute the first derivative of the marginal log-likelihood approximation in the poisson regime (when gamma -> inf).

    ARGS:
        model - instance of Model class
    RETURNS:
        jax.numpy array; (n_targs,)
    """
    summands = (y_i - mu_i)**2 - y_i
    return -jnp.sum((1 / (2 * disp_i**2)) * summands).squeeze()

def score_disp(model, const: float = 1e-4) -> jnp.array:
    r"""
    Compute the first derivative of the marginal log-likelihood with respect to dispersion parameter. Checks whether a target's estimated dispersion > cutoff and uses the appropriate regime. 

    ARGS:
        model - instance of Model class
    RETURNS:
        jax.numpy array; (n_targs, 1)
    """
    score_func = {
        'nb': score_disp_i_nb,
        'poi': score_disp_i_poi
    }
    y_arr = model.data.construct_y()
    mu_arr = intermed.compute_delta(model, inv = False)
    disp_arr = model.disp_arr

    ## cut-off value for flipping to Poisson regime
    cutoff = helpers.est_prelim_cutoff(model.compute_mu(), const)
    disp_regs = [
        'poi' if model.disp_arr[i] > cutoff else 'nb' for i in range(model.data.n_targs)
    ]
    scores = jnp.array([
        score_func[reg](y_i, mu_i, disp_i) for reg, y_i, mu_i, disp_i in zip(disp_regs, y_arr, mu_arr, disp_arr)
    ])
    return jnp.expand_dims(scores, 1)
    
##----------------------------------------------------------------------------##
## Working score for variance components.                                     ##
##----------------------------------------------------------------------------##
def work_score_var_comp(
        model
    ):
    r"""
    Compute the working score, the first derivative of the (full) marginal quasi-log-likelihood with respect to variance component vector evaluated at the provided parameter values. Equation 9 in Lin (1997).

    ARGS:
        model - instance of Model class
    RETURNS:
        jax.numpy array  (n_targs, q)
    """
    q = model.data.z_arr.shape[-1]
    n_targs = model.data.n_targs
    scores = np.zeros((n_targs, q))

    ## initial computations
    resid_vals     = model.compute_resid()
    xi_vals        = intermed.compute_xi(model)
    delta_inv_omega_vals = intermed.compute_delta_inv_omega(model)
    for j in range(q):
        z_j = model.data.z_arr[:, j]

        ## vectorize wrapper --- compute target-by-target
        def wrap_func(resid_i, delta_inv_omega_i, xi_i):
            ## upsilon_i
            upsilon_i = resid_i * delta_inv_omega_i
            term1 = upsilon_i * z_j
            term2 = xi_i * z_j**2
            return jnp.sum(term1)**2 - jnp.sum(term2)
    
        ## compute values
        vmap_func = jax.vmap(wrap_func, in_axes = [0, 0, 0])
        summands = vmap_func(resid_vals, delta_inv_omega_vals, xi_vals)
        scores[:, j] = summands
    return jnp.array(scores)



##----------------------------------------------------------------------------##
## Orthogonalized score for variance components.                              ##
##----------------------------------------------------------------------------##
def ortho_score_dict(model):
    """
    Wrapper to compute necessary quantities for computing the orthogonalized score and its covariance matrix. Helpful so we only have to compute these large arrays one time.

    ARGS: model - instance of Model class
    RETURNS: a dictionary containing the working score for the variance components, the nuisance score components, the nuisance parameter information blocks, and the population derivative blocks.
    """
    ## working score for variance components 
    work_score = work_score_var_comp(model) # (n_targs, q)
    ## covariance
    work_info = info.info_var_comp_var_comp(model) # (n_targs, q, q)

    ## nuisance score components
    score_f = score_fx_effs(model) # (n_targs, p)
    score_d = score_disp(model)    # (n_targs, 1)

    ## nuisance information blocks
    info_f = info.info_fx_effs(model) # (n_targs, p, p)
    info_d = info.info_disp(model)    # (n_targs, 1, 1)

    ## covariance of working score and nuisance score blocks
    info_wk_f = info.info_var_comp_fx_effs(model) # (n_targs, q, p)
    info_wk_d = info.info_var_comp_disp_nb(model) # (n_targs, q, 1)
    
    ## pop deriv blocks
    pd_f = info.correct_fx_effs(model) # (n_targs, p, q)
    pd_d = info.correct_disp(model)    # (n_targs, 1, q)
    return {
        'wk_score': work_score,
        'f_score': score_f,
        'd_score': score_d,
        'wk_info': work_info,
        'f_info': info_f,
        'd_info': info_d,
        'wk_f_info': info_wk_f,
        'wk_d_info': info_wk_d,
        'f_pd': pd_f,
        'd_pd': pd_d
    }

def ortho_score_var_comp(
        score_dict: dict,
        scale: bool = True,
        rtol: float = 1e-10,
        atol: float = 1e-14
    ):
    r"""
    Compute the Neyman-orthogonalized score vector for the variance component. 

    ARGS:
        model - instance of Model class
        scale - boolean; whether to scale by :math:`\frac{1}{\sqrt{k}}`. default: True
    RETURNS:
        jax.numpy array; the first derivative of the full data's marginal log-likelihood with respect to variance component vector (q, 1)
        float; scaling value (1 if scale == False)
    RAISES:
        NONE
    """
    work_score, score_f, score_d, _, info_f, info_d, info_wk_f, _, pd_f, pd_d = score_dict.values()
    q = info_wk_f.shape[1] ## pd_f is (n_targs, p, q)
    scores = np.zeros((q, 1))

    ## loop over variance components elements
    for j in range(q):
        ## subset to variance component element j
        work_score_j = work_score[:, j]
        pd_f_j = pd_f[:, :, j]
        pd_d_j = pd_d[:, :, j]

        ## target i orthogonalization
        def ortho_i(work_score_j_i, pd_f_j_i, info_f_i, score_f_i,  pd_d_j_i, info_d_i, score_d_i):
            ## fixed effects correction
            if helpers.check_stabil(info_f_i, rtol, atol):
                crct_f = pd_f_j_i.T @ jnp.linalg.solve(info_f_i, score_f_i)
            else:
                crct_f = 0.0
            ## dispersion correction
            if helpers.check_stabil(info_d_i, rtol, atol):  
                crct_d = pd_d_j_i.T @ jnp.linalg.solve(info_d_i, score_d_i)
            else:
                crct_d = 0.0
            return work_score_j_i - crct_f - crct_d
        
        ortho_targs = jnp.array(
            [ortho_i(work_score_j_i, pd_f_j_i, info_f_i, score_f_i, pd_d_j_i, info_d_i, score_d_i) for work_score_j_i, pd_f_j_i, info_f_i, score_f_i, pd_d_j_i, info_d_i, score_d_i in zip(work_score_j, pd_f_j, info_f, score_f, pd_d_j, info_d, score_d)]
        )
        scores[j, :] = jnp.sum(ortho_targs)

    if scale:
        ## 2-norm
        scale_val = jnp.linalg.norm(scores)
    else:
        scale_val = 1

    return jnp.array(scores) / scale_val, scale_val



##----------------------------------------------------------------------------##
## Wrappers.                                                                  ##
##----------------------------------------------------------------------------##
def score_nuisance(model) -> jnp.array:
    r"""
    Wrapper to compute the score vector for the nuisance parameters (fx_effs, disp).

    ARGS:
        model - instance of Model class
    RETURNS:
        jax.numpy array; (n_targs, p + 1)
    RAISES:
        NONE
    """
    sc_fx_effs = score_fx_effs(model)
    sc_disp    = score_disp(model)
    return jnp.concatenate([sc_fx_effs, sc_disp], axis = 1)








################################################################################

# def compute_score_fx_effs(model) -> jnp.array:
#     r"""Compute the first derivative of the marginal log-likelihood with respect to fixed effects. 

#     ARGS:
#         model - instance of Model class
#     RETURNS:
#         jax.numpy array; (n_targs, p)
#     RAISES:
#         NONE
#     """
#     omega_delta_vals = intermed.compute_delta_inv_omega(model)
#     resid_vals = model.compute_resid()
#     upsilon_vals = omega_delta_vals * resid_vals
#     return upsilon_vals @ model.data.x_arr


# def compute_score_disp(model):
#     """
#     Wrapper for helpers.score_disp_i(). Applies function target-wise to data.

#     ARGS: model
#     RETURNS: np.array - (n_targs, 1)
#     """
#     def score_disp_i(y_i, mu_i, disp_i):
#         dg_term = helpers.digamma_diff(y_i, disp_i)
#         log_1p_term = jnp.log1p(mu_i / disp_i)
#         resid_term = (y_i - mu_i) / (disp_i + mu_i)
#         return jnp.sum(dg_term - log_1p_term - resid_term)

#     y_arr = model.data.construct_y()
#     mu_arr = intermed.compute_delta(model, inv = False)
#     disp_arr = model.disp_arr
#     return jnp.expand_dims(jnp.stack([score_disp_i(y_i, mu_i, disp_i) for y_i, mu_i, disp_i in zip(y_arr, mu_arr, disp_arr)], axis = 0), 1)


# def compute_score_nuisance(model) -> jnp.array:
#     r"""Wrapper to compute the first derivative of the marginal log-likelihood with respect to nuisance (fx effs and dispersion) parameters. 

#     ARGS:
#         model - instance of Model class
#     RETURNS:
#         jax.numpy array; (n_targs, p + 1)
#     RAISES:
#         NONE
#     """
#     score_fx_effs = compute_score_fx_effs(model)
#     score_disp    = compute_score_disp(model)
#     return jnp.concatenate([score_fx_effs, score_disp], axis = 1)



# def compute_eff_score(
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
#     resid_vals           = model.compute_resid()
#     xi_vals              = intermed.compute_xi(model)
#     delta_inv_omega_vals = intermed.compute_delta_inv_omega(model)
#     ## (n_targs, p + 1, p + 1)
#     info_nn_inv = jnp.linalg.pinv(info.info_nuisance_nuisance(model))
#     ## (n_targs, p + 1, 1)
#     score_n = jnp.expand_dims(compute_score_nuisance(model), 2)
#     ## (n_targs, p + 1, q)
#     info_vcn = info.info_var_comp_nuisance(model)

#     for idx in range(q):
#         ## (n_samps, n_samps) --- same for all targets
#         a_j_arr = intermed.compute_a_j(model, j = idx)
#         info_vcn_q = info_vcn[:, :, idx]

#         ## vectorize wrapper --- compute target-by-target
#         def wrap_func(resid_targ, delta_inv_omega_targ, xi_targ, vcn_i, nn_inv_i, s_n_i):
#             rdo_targ = resid_targ * delta_inv_omega_targ
#             term1 = rdo_targ.T @ a_j_arr @ rdo_targ
#             term2 = jnp.diag(xi_targ) @ a_j_arr
#             crct = vcn_i.T @ nn_inv_i @ s_n_i
#             return (1 / 2) * (term1 - jnp.trace(term2)) - crct

#         ## compute values
#         vmap_func = jax.vmap(wrap_func, in_axes = [0, 0, 0, 0, 0, 0])
#         summands = vmap_func(resid_vals, delta_inv_omega_vals, xi_vals, info_vcn_q, info_nn_inv, score_n)
#         full_log_lik = jnp.sum(summands)
#         scores = scores.at[idx].set(full_log_lik)
#     if scale:
#         ## 2-norm
#         scale_val = jnp.linalg.norm(scores)
#     else:
#         scale_val = 1

#     wrt_var_comp = scores / scale_val

#     return wrt_var_comp, scale_val



# def compute_score_psi_adj_fx_effs(model) -> jnp.array:
#     r"""
#     Compute the first stage of the score orthogonalization.

#     ARGS:
#         model - instance of Model class
#     RETURNS:
#         jax.numpy array; (n_targs, q)
#     RAISES:
#         NONE
#     """
#     score_fx_effs = compute_score_fx_effs(model)
#     info_fx_effs = info.info_fx_effs_fx_effs(model)
#     q = model.data.z_arr.shape[-1]
#     scores = jnp.zeros((model.data.n_targs, q))

#     ## initial computations
#     resid_vals           = model.compute_resid()
#     xi_vals              = intermed.compute_xi(model)
#     delta_inv_omega_vals = intermed.compute_delta_inv_omega(model)

#     for idx in range(q):
#         ## (n_targs, p)
#         psi_j_fx_effs = info.info_var_comp_j_fx_effs(model, j = idx)
#         ## (n_samps, n_samps) --- same for all targets
#         a_j_arr = intermed.compute_a_j(model, j = idx)
#         ## vectorize wrapper --- compute target-by-target
#         def wrap_func(resid_targ, delta_inv_omega_targ, xi_targ, cov_psi_j_fx_effs_i, var_fx_effs_i, score_fx_effs_i):
#             rdo_targ = resid_targ * delta_inv_omega_targ
#             term1 = rdo_targ.T @ a_j_arr @ rdo_targ
#             term2 = jnp.diag(xi_targ) @ a_j_arr
#             score_j = term1 - jnp.trace(term2)
#             return score_j - cov_psi_j_fx_effs_i.T @ jnp.linalg.solve(var_fx_effs_i, score_fx_effs_i)
#         ## compute values
#         vmap_func = jax.vmap(wrap_func, in_axes = [0, 0, 0, 0, 0, 0])
#         psi_j_score = vmap_func(resid_vals, delta_inv_omega_vals, xi_vals, psi_j_fx_effs, info_fx_effs, score_fx_effs)
#         scores = scores.at[:, idx].set(psi_j_score)
#     return scores



# def compute_score_var_comp(
#         model,
#         scale: bool = True
#     ):
#     r"""
#     Compute the Neyman-orthogonalized score vector for the variance component. 

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