"""Computations related to information matrix."""

import jax
import itertools

import jax.numpy    as jnp
import numpy        as np
import scipy.sparse as sps

from stat_module import intermed


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
        j           - int; first index of tau sqr score
    RETURNS:
        jnp.array - the covariance of score for fixed effects vector and j-th coordinate of var_comp (n_targs * p,)
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
    ## for each row in x_arr (p, 1)
    def wrap_func(omega_i):
        diag_omega = jnp.diag(jnp.squeeze(omega_i))
        return jnp.matmul(jnp.matmul(x_arr.T, diag_omega), x_arr)
    vmap_func = jax.vmap(wrap_func, in_axes = [0])
    ## (n_samps, p, p)
    vals = vmap_func(omega_vals)
    return vals

    # n_targs = model.data.n_targs
    # n_samps = model.data.n_samps
    # targs_samps = n_targs * n_samps
    # ## omega values to diagonal matrix (n_targs * n_samps, n_targs * n_samps)
    # omega_vals = intermed.compute_omega(model)
    # omega_crc  = sps.coo_array(
    #     (jnp.ravel(omega_vals), (range(targs_samps), range(targs_samps))), 
    #     (targs_samps, targs_samps)
    # ).tocsc().todense()
    # ## (n_samps, p)
    # x_arr = model.data.x_arr
    # ## x_arr values to sparse matrix (n_targs * n_samps, n_targs * p)
    # vals = []
    # row_idx = []
    # col_idx = []
    # ## loop shouldn't be prohibitively slow maybe? because p should be like 1 or 2
    # for i in range(x_arr.shape[-1]):
    #     x_arr_i  = x_arr[:, i]
    #     vals    += list(itertools.chain.from_iterable(itertools.repeat(x_arr_i, n_targs)))
    #     row_idx += list(range(targs_samps))
    #     col_idx += list(np.repeat(range(n_targs), n_samps) + n_targs * i)
    # ## each column is (targs_samps,) long but only n_targs coordinates are non-zero
    # ## repeats each column of x_arr n_targs times to account for the fact that
    # ## the first n_targs entries on the diagonal of omega_vals come from sample 1
    # ## the second come from sample 2, etc.
    # x_arr_crc = sps.coo_array(
    #     (list(vals), (list(row_idx), list(col_idx))),
    #     (targs_samps, n_targs * x_arr.shape[-1])
    # ).tocsc().todense()
    # big_mat = x_arr_crc.T @ omega_crc @ x_arr_crc
    # return  big_mat


def compute_info(
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
    q = model.data.q
    ## initializations
    p = model.data.x_arr.shape[1]
    info_fx_var = jnp.zeros((model.data.n_targs, p, q))
    info_var = jnp.zeros((q, q))

    ## fx effs
    info_fx_effs = info_fx_effs_fx_effs(model)
    # info_arr = info_arr.at[0:total_p, 0:total_p].set(info_fx_effs)

    for idx1 in range(q):
        info_fx_tau_j = info_fx_effs_var_comp_j(model, j = idx1)
        info_fx_var = info_fx_var.at[:, :, idx1].set(info_fx_tau_j)

        for idx2 in range(idx1 + 1):
            info_tau_tau = info_var_comp_j_var_comp_l(
                model,
                j = idx1,
                l = idx2
            )
            info_var = info_var.at[idx1, idx2].set(info_tau_tau)
            info_var = info_var.at[idx2, idx1].set(info_tau_tau)

    if scale_val is not None:
        info_fx_effs /= scale_val**2 
        info_fx_var  /= scale_val**2
        info_var     /= scale_val**2
    return info_fx_effs, info_fx_var, info_var



