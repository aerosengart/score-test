"""Computations related to information matrix."""

import jax
import itertools
import scipy

import jax.numpy    as jnp
import numpy        as np
import scipy.sparse as sps


from stat_module import intermed



def info_fx_effs_fx_effs(
        model
    ) -> jnp.array:
    """
    Compute variance of score for fixed effects vector. Eq. (15) in Lin (1997).

    ARGS:
        model       - an instance of class Model
    RETURNS:
        jnp.array - the covariance matrix for the score components for the fixed effects (n_targs, p, p)
    RAISES:
        None
    """
    ## omega values (rel_targs, n_samps)
    omega_vals = intermed.compute_omega(model)#[model.data.rel_targs, :]
    ## (n_samps, p)
    x_arr = model.data.x_arr
    vals = jnp.swapaxes(x_arr[None, :, :] * omega_vals[:, :, None], 1, 2) @ x_arr
    return vals

def info_disp_disp(
        model, 
        tol: float = 1e-12
    ) -> jnp.array:
    """
    Compute variance of score for dispersion parameters. 

    ARGS:
        model  - an instance of class Model
        tol    - positive float
    RETURNS:
        jnp.array - the covariance matrix for the score components for the dispersion parameters (n_targs, 1, 1)
    RAISES:
        None
    """
    ## can precompute trigamma expectation
    mu_arr = model.compute_mu()
    disp_arr = model.disp_arr
    probs = disp_arr / (disp_arr + mu_arr)
    max_vals = scipy.stats.nbinom.ppf(1.0 - tol, disp_arr, probs)
    max_vals = np.array(np.ceil(max_vals), dtype = 'int')

    def tg_exp_ij(max_val, disp, prob):
        range_vals = np.arange(max_val)
        ps = scipy.stats.nbinom.sf(range_vals, disp, prob)
        vals = 1 / (disp + np.array(range_vals))**2
        return -np.sum(vals * ps)

    tg_exp_v = np.vectorize(tg_exp_ij)
    tg_exps = tg_exp_v(max_vals, disp_arr * np.ones((1, model.data.n_samps)), probs)
    return -np.sum(tg_exps + mu_arr / (disp_arr * (disp_arr + mu_arr)), axis = 1)


def info_var_comp_j_var_comp_l(
        model,
        j: int = 0,
        l: int = 0
    ) -> jnp.array:
    """
    Compute variance of score for var_comp. 

    ARGS:
        model - an instance of class Model
        j     - int; first index of tau sqr score
        l     - int; second index of tau sqr score
    RETURNS:
        jnp.array - the covariance of the score component corresponding to the j-th and l-th coordinates of var_comp (1, 1)
    RAISES:
        None
    """
    ## (n_targs, n_samps, n_samps)
    r_arrs = intermed.compute_r(model)
    ## (n_samps, n_samps)
    a_j_arr = intermed.compute_a_j(model, j = j)
    a_l_arr = intermed.compute_a_j(model, j = l)

    def wrap_func(r_arr):
        return jnp.sum(jnp.multiply(a_j_arr, jnp.multiply(r_arr, a_l_arr)))

    vmap_func = jax.vmap(wrap_func, in_axes = [0])
    vals = vmap_func(r_arrs)
    return jnp.sum(vals) / 4

def info_fx_effs_var_comp_j(
        model,
        j: int
    ) -> jnp.array:
    """
    Compute covariance of score for fixed effects vector and j-th coordinate of var_comp. Eq. (14) in Lin (1997).

    ARGS:
        model       - an instance of class Model
        j           - int; index of variance components score
    RETURNS:
        jnp.array - the covariance of score for fixed effects vector and j-th coordinate of var_comp (n_targs, p)
    RAISES:
        None
    """
    ## (rel_targs, n_samps)
    c_vals = intermed.compute_c(model)#[model.data.rel_targs, :]
    ## (n_samps, n_samps)
    a_j_arr = intermed.compute_a_j(model, j = j)
    ## (n_samps, 1)
    a_j_diag = jnp.expand_dims(jnp.diag(a_j_arr), 1)
    ## (n_samps, p)
    x_arr = model.data.x_arr
    def wrap_func(c_targ):
        c_targ = jnp.expand_dims(c_targ, 1)
        return a_j_diag * c_targ * x_arr 
    vmap_func = jax.vmap(wrap_func, in_axes = [0])
    ## (rel_targs, n_samps, p)
    vals = vmap_func(c_vals)
    sum_vals = jnp.sum(vals, 1)
    ## unravel Fortran style --- column-major because we want to do the values
    ## for the first covariate, then the second covariate, etc.
    # result = jnp.ravel(sum_vals, order = 'F')
    ## unravel C style --- row-major because we want to do the values
    ## for the first target, then the second target, etc.
    return sum_vals #jnp.ravel(sum_vals, order = 'C')

def info_fx_effs_var_comp(model):
    """
    Compute covariance of score for fixed effects vector and variance component vector.

    ARGS:
        model       - an instance of class Model
    RETURNS:
        jnp.array - (n_targs, q, p)
    RAISES:
        None
    """
    omega_vals = intermed.compute_omega(model)
    z_sqr = model.data.z_arr**2
    return (1 / 2) * np.swapaxes(z_sqr[None, :, :] * omega_vals[:, :, None], 1, 2) @ model.data.x_arr

def info_disp_var_comp(model) -> jnp.array:
    """
    Compute covariance of score for dispersion parameters and j-th coordinate of var_comp.

    ARGS:
        model       - an instance of class Model
        j           - int; index of variance components score
    RETURNS:
        jnp.array - the covariance of score for variance components and j-th coordinate of var_comp (n_targs, q, 1)
    RAISES:
        None
    """
    z_sqr = model.data.z_arr**2
    phi_vals = (model.compute_mu() / (model.disp_arr + model.compute_mu()))**2
    return -(1 / 2) * np.swapaxes(z_sqr, 0, 1)[None, :, :] @ phi_vals[:, :, None]


def compute_info_eff(
        model,
        scale_val: float = None
    ) -> jnp.array:
    """
    Compute the (scaled) information matrix under H0. 

    ARGS:
        model       - an instance of class Model
        scale_val   - float; scaling value used to scale the score vector
    RETURNS:
        three jax.numpy arrays; the upper left block of the information matrix under H0 corresponding to the fixed effects covariances (p*, p*), the rightmost "rectangle" block corresponding to the covariances of the fixed effects and variance components (p*, q), the lower right block corresopnding to the variance components covariances (q, q)
    RAISES: None
    """
    ## initializations
    q = model.data.z_arr.shape[-1]
    info_var = jnp.zeros((q, q))

    ## fx effs
    info_fx      = info_fx_effs_fx_effs(model)
    info_disp    = info_disp_disp(model)
    info_fx_vc   = info_fx_effs_var_comp(model)
    info_disp_vc = info_disp_var_comp(model)

    for idx1 in range(q):
        for idx2 in range(idx1 + 1):
            info_tau_tau = info_var_comp_j_var_comp_l(
                model,
                j = idx1,
                l = idx2
            )
            info_var = info_var.at[idx1, idx2].set(info_tau_tau)
            info_var = info_var.at[idx2, idx1].set(info_tau_tau)

    ## variance of efficient score
    try:
        info_fx_crct = info_fx_vc @ np.linalg.solve(info_fx, np.swapaxes(info_fx_vc, 1, 2))
    except:
        info_fx_crct = info_fx_vc @ np.linalg.pinv(info_fx) @ np.swapaxes(info_fx_vc, 1, 2)
    ## only correct those with finite dispersion
    mask = ~np.isinf(model.disp_arr).squeeze()
    info_disp_crct = np.zeros((model.data.n_targs, model.data.z_arr.shape[-1], model.data.z_arr.shape[-1]))
    d_crct_vals = info_disp_vc[mask] @ np.linalg.solve(info_disp[mask][:, None, None], np.swapaxes(info_disp_vc[mask], 1, 2))
    info_disp_crct[mask] = d_crct_vals
    #info_disp_crct = info_disp_vc @ np.linalg.solve(info_disp[:, None, None], np.swapaxes(info_disp_vc, 1, 2))
    info_eff = info_var - info_fx_crct.sum(axis = 0) - info_disp_crct.sum(axis = 0)

    if scale_val is not None:
        info_eff  /= scale_val**2
    return info_eff











# def compute_info(
#         model,
#         scale_val: float = None
#     ) -> jnp.array:
#     """
#     Compute the (scaled) information matrix under H0. 

#     ARGS:
#         model       - an instance of class Model
#         scale_val   - float; scaling value used to scale the score vector
#     RETURNS:
#         three jax.numpy arrays; the upper left block of the information matrix under H0 corresponding to the fixed effects covariances (p*, p*), the rightmost "rectangle" block corresponding to the covariances of the fixed effects and variance components (p*, q), the lower right block corresopnding to the variance components covariances (q, q)
#     RAISES: None
#     """
#     q = model.data.z_arr.shape[-1]
#     ## initializations
#     p = model.data.x_arr.shape[-1]
#     info_fx_var = jnp.zeros((model.data.n_targs, p, q))
#     info_var = jnp.zeros((q, q))

#     ## fx effs
#     info_fx_effs = info_fx_effs_fx_effs(model)
#     # info_arr = info_arr.at[0:total_p, 0:total_p].set(info_fx_effs)

#     for idx1 in range(q):
#         info_fx_tau_j = info_fx_effs_var_comp_j(model, j = idx1)
#         info_fx_var = info_fx_var.at[:, :, idx1].set(info_fx_tau_j)

#         for idx2 in range(idx1 + 1):
#             info_tau_tau = info_var_comp_j_var_comp_l(
#                 model,
#                 j = idx1,
#                 l = idx2
#             )
#             info_var = info_var.at[idx1, idx2].set(info_tau_tau)
#             info_var = info_var.at[idx2, idx1].set(info_tau_tau)

#     if scale_val is not None:
#         info_fx_effs /= scale_val**2 
#         info_fx_var  /= scale_val**2
#         info_var     /= scale_val**2
#     return info_fx_effs, info_fx_var, info_var



