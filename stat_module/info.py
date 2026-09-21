"""Computations related to information matrix."""

import jax
import itertools

jax.config.update("jax_enable_x64", True)

import jax.numpy    as jnp
import numpy        as np
import scipy.sparse as sps

import scipy

from stat_module import intermed
from stat_module import helpers


##----------------------------------------------------------------------------##
## Information block for fixed effects by fixed effects.                      ##
## Derived as expectation of fixed effects score vector.                      ##
##----------------------------------------------------------------------------##
def info_fx_effs(
        model
    ) -> jnp.array:
    """
    Compute variance of score for fixed effects vector. 

    This the same as when calculating from _true_ marginal log-likelihood. 

    TODO: could add poisson version where we just use diag(mu) instead of diag(omega)

    ARGS:
        model       - an instance of class Model
    RETURNS:
        jnp.array - the covariance matrix for the score components for the fixed effects (n_targs, p, p)
    RAISES:
        None
    """
    ## omega values (n_targs, n_samps)
    omega_vals = intermed.compute_omega(model)
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


##----------------------------------------------------------------------------##
## Information block for dispersion parameters by dispersion parameters.      ##
## Derived as negative expected Hessian of log-likelihood.                    ##
## Includes regime switch based upon magnitude of dispersion parameter(s).    ##
##----------------------------------------------------------------------------##
def info_disp_i_nb(mu_i, disp_i, tol = 1e-12):    
    r"""
    Compute variance of score for the dispersion parameter vector in the negative binomial regime (dispersion parameterized by gamma not too large).

    ARGS:
        TODO
    RETURNS:
        jax.numpy array; (1,)
    """
    ## E[trigamma(y + disp) - trigamma(disp)]
    tg_term = helpers.trigamma_diff_exp(mu_i, disp_i, tol)
    term2 = mu_i / (disp_i * (disp_i + mu_i))
    return -jnp.sum(tg_term + term2)

def info_disp_i_poi(mu_i, disp_i, tol = 1e-12):
    r"""
    Compute variance of score for the dispersion parameter vector in the poisson regime (dispersion parameterized by gamma very large).

    ARGS:
        TODO
    RETURNS:
        jax.numpy array; (1,)
    """
    #return (1 / 2) * jnp.sum(mu_i**2)
    return (1 / 2) * jnp.sum(mu_i**2 / disp_i**4)

def info_disp(model, const = 1e-4, tol = 1e-12):
    """
    Calls the appropriate one of above based upon cutoff and estimated dispersion parameter.

    ARGS:
        TODO
    RETURNS:
        jax.numpy array; (n_targs, 1, 1)
    """
    info_func = {
        'nb': info_disp_i_nb,
        'poi': info_disp_i_poi
    }
    mu_arr = intermed.compute_delta(model, inv = False)
    disp_arr = model.disp_arr

    ## cut-off value for flipping to Poisson regime
    cutoff = helpers.est_prelim_cutoff(model.compute_mu(), const)
    disp_regs = [
        'poi' if model.disp_arr[i] > cutoff else 'nb' for i in range(model.data.n_targs)
    ]
    infos = jnp.array([
        info_func[reg](mu_i, disp_i, tol) for reg, mu_i, disp_i in zip(disp_regs, mu_arr, disp_arr)
    ])
    return jnp.expand_dims(infos, [1, 2])


##----------------------------------------------------------------------------##
## Population derivatives for fixed effects.                                  ##
##----------------------------------------------------------------------------##
def correct_j_fx_effs(
        model,
        j: int = 0
    ) -> jnp.array:
    """
    Derivative of expected score for j-th element of variance component vector with respect to fixed effects for target i. 

    ARGS:
        model       - an instance of class Model
        j           - int; first index of tau sqr score
    RETURNS:
        jnp.array (n_targs, p)
    """
    x_arr = model.data.x_arr
    z_j = model.data.z_arr[:, j]
    z_j_sqr = z_j**2
    omega_vals = intermed.compute_omega(model)
    def wrap_func(omega_i):
        omega_diag = jnp.diag(omega_i)
        return -(1 / 2) * z_j_sqr.T @ omega_diag @ x_arr
    vmap_func = jax.vmap(wrap_func, in_axes = [0])
    ## (n_targs, p)
    vals = vmap_func(omega_vals)
    return vals


##----------------------------------------------------------------------------##
## Population derivatives for dispersion parameters.                          ##
## Includes regime switch based upon magnitude of dispersion parameter(s).    ##
##----------------------------------------------------------------------------##
def correct_j_disp_i_nb(
        phi_i,
        z_j
    ) -> jnp.array:
    """
    ARGS:
        model       - an instance of class Model
        j           - int; first index of tau sqr score
    RETURNS:
        jnp.array - the covariance of score for j-th coordinate of var_comp and fixed effects vector (1,)
    RAISES:
        None
    """
    z_j_sqr = z_j**2
    val =  (1 / 2) * z_j_sqr.T @ phi_i
    return val

def correct_j_disp_i_poi(
        omega_i,
        z_j
    ) -> jnp.array:
    """
    ARGS:
        model       - an instance of class Model
        j           - int; first index of tau sqr score
    RETURNS:
        jnp.array - the covariance of score for j-th coordinate of var_comp and fixed effects vector (1,)
    RAISES:
        None
    """
    n_samps = len(z_j)
    z_j_sqr = z_j**2
    omega_diag = jnp.diag(omega_i)
    val = -(1 / 2) * z_j_sqr.T @ omega_diag @ jnp.ones((n_samps))
    return val

def correct_j_disp(model, j: int = 0, const: float = 1e-4):
    """
    Calls the appropriate one of above based upon cutoff and estimated dispersion parameter.

    ARGS:
        TODO
    RETURNS:
        jax.numpy array; (n_targs, 1)
    """
    info_func = {
        'nb': correct_j_disp_i_nb,
        'poi': correct_j_disp_i_poi
    }
    omega_vals = intermed.compute_omega(model)
    phi_vals = intermed.compute_phi(model)
    disp_arr = model.disp_arr
    
    n_targs = model.data.n_targs
    q = model.data.z_arr[-1]

    ## cut-off value for flipping to Poisson regime
    cutoff = helpers.est_prelim_cutoff(model.compute_mu(), const)
    disp_regs = [
        'poi' if disp_arr[i] > cutoff else 'nb' for i in range(n_targs)
    ]
    ## intermed vals used in computations -- omega or phi depending on nb or poi
    vals = [
        omega_vals[i] if disp_regs[i] == 'poi' else phi_vals[i] for i in range(n_targs)
    ]
    z_j = model.data.z_arr[:, j]
    # crcts = jnp.array([
    #         info_func[reg](vals_i, z_j) for reg, vals_i in zip(disp_regs, vals)
    #     ])
    crcts = jnp.array([
        correct_j_disp_i_nb(phi_i, z_j) for phi_i in phi_vals
    ])
    return jnp.expand_dims(crcts, 1)



##----------------------------------------------------------------------------##
## Covariance of working score and nuisance score.                            ##
##----------------------------------------------------------------------------##
def info_var_comp_fx_effs(model):
    """
    Compute the covaraince of the working score for the variance components and the score for the fixed effects.

    ARGS:
        model       - an instance of class Model
        j           - int; first index of tau sqr score
    RETURNS:
        jnp.array (n_targs, q, p)
    """
    x_arr = model.data.x_arr
    z_arr = model.data.z_arr
    p = x_arr.shape[-1]
    q = z_arr.shape[-1]
    n_targs = model.data.n_targs
    omega_vals = intermed.compute_omega(model)
    info_vc_f = np.empty((n_targs, q, p))

    for j in range(q):
        z_j_sqr = z_arr[:, j]**2
        def targ_info_vc_f(omega_i):
            omega_z = jnp.expand_dims(omega_i * z_j_sqr, 1)
            omega_z_x = jnp.einsum('ij,ik->jk', omega_z, x_arr)
            return (1 / 2) * omega_z_x

        vmap_info_vc_f = jax.vmap(targ_info_vc_f, in_axes = [0])
        info_vc_f[:, j, :] = jnp.squeeze(vmap_info_vc_f(omega_vals))
    return jnp.array(info_vc_f)

def info_var_comp_j_disp_i_nb(mu_i, disp_i, omega_i, delta_inv_omega_i, z_j, tol = 1e-12):
    """
    Compute the covariance of the j-th working score for the variance components and the score for the dispersion parameter for target i in the negative binomial regime.

    ARGS:
        model       - an instance of class Model
        j           - int; first index of tau sqr score
    RETURNS:
        jnp.array (1,)
    """
    ## initial computations
    dg_diff_exp = helpers.digamma_diff_exp(mu_i, disp_i, tol)
    dg_y_ctr_exp = helpers.digamma_y_ctr_sqr_exp(mu_i, disp_i, tol)
    dg_y_ctr_sqr_exp = helpers.digamma_y_ctr_sqr_exp(mu_i, disp_i, tol)
    ## E[dg(y + disp) - dg(disp)] + dg(disp) = E[dg(y + disp)]
    dg_exp = dg_diff_exp + scipy.special.digamma(disp_i)
    z_sqr = z_j**2
    ## first and last summation
    term1 = jnp.sum(delta_inv_omega_i**2 * (dg_y_ctr_sqr_exp - omega_i * dg_exp) * z_sqr)
    term3 = jnp.sum((omega_i / (disp_i + mu_i)) * (dg_y_ctr_exp + (mu_i / (disp_i + mu_i))) * z_sqr)
    ## second summation is a double sum over non-matching indices
    inner_sum_vec = (omega_i - omega_i * delta_inv_omega_i**2) * z_sqr
    outer_prod = jnp.outer(dg_exp, inner_sum_vec)
    term2 = jnp.sum(jnp.fill_diagonal(outer_prod, 0, inplace = False))
    return (1 / 2) * (term1 + term2 - term3)

def info_var_comp_j_disp_i_poi(mu_i, disp_i, omega_i, delta_inv_omega_i, z_j, tol = 1e-12):
    """
    Compute the covariance of the j-th working score for the variance components and the score for the dispersion parameter for target i in the poisson regime.

    ARGS:
        model       - an instance of class Model
        j           - int; first index of tau sqr score
    RETURNS:
        jnp.array (1,)
    """
    omega_i_sqr = omega_i**2
    z_j_sqr = z_j**2
    return (1 / 4) * jnp.sum(omega_i_sqr * z_j_sqr)

def info_var_comp_j_disp(model, j: int = 0, const: float = 1e-4):
    """
    Calls the appropriate one of above based upon cutoff and estimated dispersion parameter.

    ARGS:
        TODO
    RETURNS:
        jax.numpy array; (n_targs, 1)
    """
    info_func = {
        'nb': info_var_comp_j_disp_i_nb,
        'poi': info_var_comp_j_disp_i_poi
    }
    mu_arr = model.compute_mu()
    disp_arr = model.disp_arr
    omega_vals = intermed.compute_omega(model)
    delta_inv_omega_vals = intermed.compute_delta_inv_omega(model)
    n_targs = model.data.n_targs

    ## cut-off value for flipping to Poisson regime
    cutoff = helpers.est_prelim_cutoff(model.compute_mu(), const)
    disp_regs = [
        'poi' if disp_arr[i] > cutoff else 'nb' for i in range(n_targs)
    ]
    z_j = model.data.z_arr[:, j]
    crcts = jnp.array([
            info_func[reg](mu_i, disp_i, omega_i, delta_inv_omega_i, z_j) for reg, mu_i, disp_i, omega_i, delta_inv_omega_i in zip(disp_regs, mu_arr, disp_arr, omega_vals, delta_inv_omega_vals)
        ])
    return jnp.expand_dims(crcts, 1)

def info_var_comp_disp_nb(model, tol = 1e-12):
    """
    Compute the covariance of the working score for the variance components and the score for the fixed effects.

    ARGS:
        model       - an instance of class Model
        j           - int; first index of tau sqr score
    RETURNS:
        jnp.array (n_targs, q, p)
    """
    z_arr = model.data.z_arr
    q = z_arr.shape[-1]
    n_targs = model.data.n_targs
    info_vc_d = np.empty((n_targs, q, 1))

    for j in range(q):
        info_vc_d[:, j, :] = info_var_comp_j_disp(model, j)
    return jnp.array(info_vc_d)
    

##----------------------------------------------------------------------------##
## Covariance of working score.                                               ##
##----------------------------------------------------------------------------##
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
        jnp.array - the covariance of the score component corresponding to the j-th and l-th coordinates of var_comp (n_targs, 1, 1)
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
    return vals / 4


def info_var_comp_var_comp(model) -> jnp.array:
    """
    Compute the (scaled) information matrix under H0. 

    ARGS:
        model       - an instance of class Model
        scale_val   - float; scaling value used to scale the score vector
    RETURNS:
        three jax.numpy arrays; the lower right block corresopnding to the variance components score covariances (q, q)
    RAISES: None
    """
    q = model.data.z_arr.shape[-1]
    info_var = jnp.zeros((model.data.n_targs, q, q))

    for idx1 in range(q):
        for idx2 in range(idx1 + 1):
            info_tau_tau = info_var_comp_j_var_comp_l(
                model,
                j = idx1,
                l = idx2
            )
            info_var = info_var.at[:, idx1, idx2].set(info_tau_tau)
            info_var = info_var.at[:, idx2, idx1].set(info_tau_tau)

    return info_var


##----------------------------------------------------------------------------##
## Wrappers.                                                                  ##
##----------------------------------------------------------------------------##
def info_nuisance(model) -> jnp.array:
    """
    Wrapper to compute the information matrix for the nuisance parameters (fx_effs, disp).

    ARGS:
        model - an instance of class Model
    RETURNS:
        jax.numpy.array - (n_targs, p + 1, p + 1)
    RAISES: None
    """
    info_ff = info_fx_effs(model) ## (n_targs, p, p)
    info_dd = info_disp(model) ## (n_targs, 1, 1)
    info_fd = jnp.zeros((model.data.n_targs, model.data.x_arr.shape[-1], 1))
    info_df = jnp.zeros((model.data.n_targs, 1, model.data.x_arr.shape[-1]))
    info_ff_fd = jnp.concatenate([info_ff, info_fd], axis = 2)
    info_fd_dd = jnp.concatenate([info_df, info_dd], axis = 2)
    info_nuisance = jnp.concatenate([info_ff_fd, info_fd_dd], axis = 1)
    return info_nuisance

def correct_fx_effs(model):
    """
    Wrapper to compute the population derivative of the variance components with respect to the fixed effects.

    ARGS:
        model - an instance of class Model
    RETURNS:
        jax.numpy.array - (n_targs, p, q)
    RAISES: None
    """
    p = model.data.x_arr.shape[-1]
    q = model.data.z_arr.shape[-1]
    crct_fx = np.empty((model.data.n_targs, p, q))
    for j in range(q):
        crct_fx[:, :, j] = correct_j_fx_effs(model, j)
    return jnp.array(crct_fx)
        
def correct_disp(model):
    """
    Wrapper to compute the population derivative of the variance components with respect to the dispersion parameters.

    ARGS:
        model - an instance of class Model
    RETURNS:
        jax.numpy.array - (n_targs, 1, q)
    RAISES: None
    """
    p = model.data.x_arr.shape[-1]
    q = model.data.z_arr.shape[-1]
    crct_d = np.empty((model.data.n_targs, 1, q))
    for j in range(q):
        crct_d[:, :, j] = correct_j_disp(model, j)
    return jnp.array(crct_d)

def info_ortho(
        score_dict: dict,
        scale_val:  float = None,
        rtol: float = 1e-10,
        atol: float = 1e-14
    ) -> jnp.array:
    """
    Compute the (scaled) information matrix for the orthogonalized score vector under H0. 

    ARGS:
        model       - an instance of class Model
        scale_val   - float; scaling value used to scale the score vector
    RETURNS: jnp.array (n_targs, q, q) - information matrix for orthogonalized score by target
    RAISES: None
    """ 
    _, _, _, info_wk, info_f, info_d, info_wk_f, info_wk_d, pd_f, pd_d = score_dict.values()

    ## since independent, can do fixed effects and dispersion separately
    def info_ortho_targ(info_wk_i, info_f_i, info_wk_f_i, pd_f_i, info_d_i, info_wk_d_i, pd_d_i):
        ## fx effects 
        if helpers.check_stabil(info_f_i, rtol, atol):
            inv_fx_term = jnp.linalg.solve(info_f_i, pd_f_i) ## (p, q)
            wk_fx_block = -2 * info_wk_f_i @ inv_fx_term  ## (q, p) @ (p, q)
            fx_block    = pd_f_i.T @ inv_fx_term ## (1, 1)
        else:
            wk_fx_block = jnp.zeros(info_wk_i.shape)
            fx_block = jnp.zeros(info_wk_i.shape)
        ## dispersion correction
        if helpers.check_stabil(info_d_i, rtol, atol):  
            inv_d_term = jnp.linalg.solve(info_d_i, pd_d_i) ## (1, q)
            wk_d_block = -2 * info_wk_d_i @ inv_d_term ## (q, 1) @ (1, q)
            d_block    = pd_d_i.T @ inv_d_term ## (1, 1)
        else:
            wk_d_block = jnp.zeros(info_wk_i.shape)
            d_block = jnp.zeros(info_wk_i.shape)
        return info_wk_i - wk_fx_block - wk_d_block + fx_block + d_block
    
    info_ortho_targs = jnp.array(
        [info_ortho_targ(info_wk_i, info_f_i, info_wk_f_i, pd_f_i, info_d_i, info_wk_d_i, pd_d_i) for info_wk_i, info_f_i, info_wk_f_i, pd_f_i, info_d_i, info_wk_d_i, pd_d_i in zip(info_wk, info_f, info_wk_f, pd_f, info_d, info_wk_d, pd_d)]
    )

    if scale_val is not None:
        return info_ortho_targs / scale_val**2
    return info_ortho_targs








# def info_disp_gamma(model):
#     """
#     Estimate information matrix for dispersion parameter (gamma) when var(y) = mu + mu^2 / gamma. Only appropriate for gamma not too large.

#     ARGS: model
#     RETURNS: np.array - (n_targs, 1, 1)
#     """
#     def info_disp_i_disp_i(mu_i, disp_i):
#         ## compute E[trigamma(y + disp) - trigamma(disp)]
#         def exp_arg(x):
#             return helpers.trigamma_diff(x, disp_i)
#         exp_term = np.asarray([helpers.nb_exp(exp_arg, mu_ij, disp_i) for mu_ij in mu_i])
#         term2 = mu_i / (disp_i * (disp_i + mu_i))
#         term_var = -jnp.sum(exp_term + term2)
#         return term_var
#     mu_arr = intermed.compute_delta(model, inv = False)
#     disp_arr = model.disp_arr
#     result = np.stack([info_disp_i_disp_i(mu_i, disp_i) for mu_i, disp_i in zip(mu_arr, disp_arr)], axis = 0)
#     return jnp.expand_dims(result, (1, 2))

# def info_disp_iota(model):
#     """
#     Estimate information matrix for dispersion parameter (gamma) when var(y) = mu + mu^2 * iota. We set iota = 1 / gamma and use when gamma is very large to make more numerically stable.

#     ARGS: model
#     RETURNS: np.array - (n_targs, 1, 1)
#     """
#     def info_disp_i_disp_i(mu_i, disp_i):
#         ## compute E[trigamma(y + disp) - trigamma(disp)]
#         def exp_arg(x):
#             return helpers.trigamma_diff(x, disp_i)
#         exp_term = np.asarray([helpers.nb_exp(exp_arg, mu_ij, disp_i) for mu_ij in mu_i])
#         term2 = mu_i / (disp_i * (disp_i + mu_i))
#         term_var = -jnp.sum(exp_term + term2)
#         return term_var
#     mu_arr = intermed.compute_delta(model, inv = False)
#     disp_arr = model.disp_arr
#     result = np.stack([info_disp_i_disp_i(mu_i, disp_i) for mu_i, disp_i in zip(mu_arr, disp_arr)], axis = 0)
#     return jnp.expand_dims(result, (1, 2))


# def info_disp_gamma(model):
#     """
#     Estimate information matrix for dispersion parameter (gamma) when var(y) = mu + mu^2 / gamma.

#     ARGS: model
#     RETURNS: np.array - (n_targs, 1, 1)
#     """
#     def info_disp_i_disp_i(mu_i, disp_i, kappa3_i):
#         ## compute E[trigamma(y + disp) - trigamma(disp)]
#         if disp_i == float('inf'):
#             ## the poisson case
#             return 0.0
#         elif disp_i > 1000:
#             ## use taylor approximation of trigamma function and moments of NB2
#             ## go to 4 terms
#             a = disp_i + mu_i
#             mom2 = mu_i + mu_i**2 / disp_i
#             mom3 = kappa3_i
#             def exp_approx(a_ij, mom2_ij, mom3_ij):
#                 return scipy.special.polygamma(1, a_ij) + (1 / 2) * mom2_ij * scipy.special.polygamma(3, a_ij) + (1 / 6) * mom3_ij * scipy.special.polygamma(4, a_ij) - scipy.special.polygamma(1, disp_i)
            
#             exp_term = np.array([exp_approx(a_ij, mom2_ij, mom3_ij) for a_ij, mom2_ij, mom3_ij in zip(a, mom2, mom3)])
#         else:
#             ## actually compute expectation via numerical integration
#             def exp_arg(x):
#                 return helpers.trigamma_diff(x, disp_i)
#             exp_term = np.asarray([helpers.nb_exp(exp_arg, mu_ij, disp_i) for mu_ij in mu_i])
#         term2 = mu_i / (disp_i * (disp_i + mu_i))
#         term_var = -jnp.sum(exp_term + term2)
#         if term_var < 0:
#             ## TODO: add warning message
#             ## force zero to avoid negative variance
#             return 0.0
#         return term_var
#     mu_arr = intermed.compute_delta(model, inv = False)
#     kappa3 = model.compute_kappa(order = 3)
#     disp_arr = model.disp_arr
#     result = np.stack([info_disp_i_disp_i(mu_i, disp_i, kappa3_i) for mu_i, disp_i, kappa3_i in zip(mu_arr, disp_arr, kappa3)], axis = 0)
#     return jnp.expand_dims(result, (1, 2))


# def info_var_comp_j_var_comp_l(
#         model,
#         j: int = 0,
#         l: int = 0
#     ) -> jnp.array:
#     """
#     Compute variance of score for var_comp. 

#     ARGS:
#         model - an instance of class Model
#         j     - int; first index of tau sqr score
#         l     - int; second index of tau sqr score
#     RETURNS:
#         jnp.array - the covariance of the score component corresponding to the j-th and l-th coordinates of var_comp (1, 1)
#     RAISES:
#         None
#     """
#     ## (n_targs, n_samps, n_samps)
#     r_arrs = intermed.compute_r(model)
#     ## (n_samps, n_samps)
#     a_j_arr = intermed.compute_a_j(model, j = j)
#     a_l_arr = intermed.compute_a_j(model, j = l)

#     def wrap_func(r_arr):
#         return jnp.sum(jnp.multiply(a_j_arr, jnp.multiply(r_arr, a_l_arr)))

#     vmap_func = jax.vmap(wrap_func, in_axes = [0])
#     vals = vmap_func(r_arrs)
#     return jnp.sum(vals) / 4




# def info_fx_effs_var_comp_j(
#         model,
#         j: int
#     ) -> jnp.array:
#     """
#     Compute covariance of score for fixed effects vector and j-th coordinate of var_comp. Eq. (14) in Lin (1997).

#     ARGS:
#         model       - an instance of class Model
#         j           - int; first index of tau sqr score
#     RETURNS:
#         jnp.array - the covariance of score for fixed effects vector and j-th coordinate of var_comp (n_targs * p,)
#     RAISES:
#         None
#     """
#     ## (rel_targs, n_samps)
#     c_vals = intermed.compute_c(model)#[model.data.rel_targs, :]
#     ## (n_samps, n_samps)
#     a_j_arr = intermed.compute_a_j(model, j = j)
#     ## (n_samps, 1)
#     a_j_diag = jnp.expand_dims(jnp.diag(a_j_arr), 1)
#     ## (n_samps, p)
#     x_arr = model.data.x_arr
#     def wrap_func(c_targ):
#         c_targ = jnp.expand_dims(c_targ, 1)
#         return a_j_diag * c_targ * x_arr 
#     vmap_func = jax.vmap(wrap_func, in_axes = [0])
#     ## (rel_targs, n_samps, p)
#     vals = vmap_func(c_vals)
#     sum_vals = jnp.sum(vals, 1)
#     ## unravel Fortran style --- column-major because we want to do the values
#     ## for the first covariate, then the second covariate, etc.
#     # result = jnp.ravel(sum_vals, order = 'F')
#     ## unravel C style --- row-major because we want to do the values
#     ## for the first target, then the second target, etc.
#     return sum_vals #jnp.ravel(sum_vals, order = 'C')


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



################################################################################


# def info_var_comp_j_fx_effs(
#         model,
#         j: int
#     ) -> jnp.array:
#     """
#     Compute covariance of score for fixed effects vector (based on marginal log-likelihood) and j-th coordinate of score for variance components based on marginal quasi-log-likelihood).

#     ARGS:
#         model       - an instance of class Model
#         j           - int; first index of tau sqr score
#     RETURNS:
#         jnp.array - the covariance of score for j-th coordinate of var_comp and fixed effects vector (n_targs, p)
#     RAISES:
#         None
#     """
#     x_arr = model.data.x_arr
#     z_j = model.data.z_arr[:, j]
#     z_j_sqr = z_j**2
#     omega_vals = intermed.compute_omega(model)
#     def wrap_func(omega_i):
#         omega_diag = jnp.diag(omega_i)
#         return (1 / 2) * z_j_sqr.T @ omega_diag @ x_arr
#     vmap_func = jax.vmap(wrap_func, in_axes = [0])
#     ## (n_targs, p)
#     vals = vmap_func(omega_vals)
#     return vals

# def info_var_comp_l_disp(model, l):
#     """
#     Construct the covariance matrix for l-th component of variance component score vector with dispersion parameter score vector.

#     ARGS: model
#     RETURNS: np.array - (n_targs, 1)
#     """
#     def info_var_comp_l_disp_i(mu_i, disp_i, omega_i, omega_delta_inv_i, kappa3_i, kappa4_i, z_l):
#         if disp_i == float('inf'):
#             ## if disp == inf, then we are in the Poisson case so can ignore cov
#             return 0.0
#         elif disp_i > 1000:
#             ## use taylor approximation of trigamma function and moments of NB2
#             ## go to 4 terms
#             a = disp_i + mu_i
#             mom2 = mu_i + mu_i**2 / disp_i
#             mom3 = kappa3_i
#             mom4 = kappa4_i + 2 * mom2**2

#             def digamma_exp(a_ij, mom2_ij):
#                 ## E[digamma(y + disp)]
#                 return scipy.special.digamma(a_ij) + (1 / 2) * scipy.special.polygamma(2, a_ij) * mom2_ij
#             def digamma_y_ctr_exp(a_ij, mom2_ij, mom3_ij):
#                 ## E[digamma(y + disp) * (y - mu)]
#                 return scipy.special.polygamma(1, a_ij) * mom2_ij + (1 / 2) * scipy.special.polygamma(2, a_ij) * mom3_ij
#             def digamma_y_ctr2_exp(a_ij, mom2_ij, mom3_ij, mom4_ij):
#                 ## E[digamma(y + disp) * (y - mu)^2]
#                 return scipy.special.digamma(a_ij) * mom2_ij + scipy.special.polygamma(1, a_ij) * mom3_ij + (1 / 2) * scipy.special.polygamma(1, a_ij) * mom4_ij
    
#             dg_exps = np.asarray([digamma_exp(a_ij, mom2_ij) for a_ij, mom2_ij in zip(a, mom2)])
#             dg_y_exps = np.asarray([digamma_y_ctr_exp(a_ij, mom2_ij, mom3_ij) for a_ij, mom2_ij, mom3_ij in zip(a, mom2, mom3)])
#             dg_y2_exps = np.asarray([digamma_y_ctr2_exp(a_ij, mom2_ij, mom3_ij, mom4_ij) for a_ij, mom2_ij, mom3_ij, mom4_ij in zip(a, mom2, mom3, mom4)])
#         else:
#             dg_exps = np.zeros(mu_i.shape[0])
#             dg_y_exps = np.zeros(mu_i.shape[0])
#             dg_y2_exps = np.zeros(mu_i.shape[0])
#             ## actually compute expectation via numerical integration
#             def digamma(x):
#                 return scipy.special.digamma(x + disp_i)
#             def digamma_y_ctr(x, mu):
#                 return scipy.special.digamma(x + disp_i) * (x - mu)
#             def digamma_y_ctr2(x, mu):
#                 return scipy.special.digamma(x + disp_i) * (x - mu)**2 
#             exp_arg  = lambda x: digamma(x)
#             for j in range(mu_i.shape[0]):
#                 mu_ij = mu_i[j]
#                 exp_arg1 = lambda x: digamma_y_ctr(x, mu_ij)
#                 exp_arg2 = lambda x: digamma_y_ctr2(x, mu_ij)
#                 dg_exps[j]    = helpers.nb_exp(exp_arg, mu_ij, disp_i)
#                 dg_y_exps[j]  = helpers.nb_exp(exp_arg1, mu_ij, disp_i)
#                 dg_y2_exps[j] = helpers.nb_exp(exp_arg2, mu_ij, disp_i)
#         sum1 = (omega_delta_inv_i**2 * (dg_y2_exps - omega_i * dg_exps) * z_l**2).sum()
#         j_vec = omega_i * (1 - omega_delta_inv_i**2) * z_l**2
#         outer_prod = jnp.fill_diagonal(jnp.outer(dg_exps, j_vec), 0, inplace = False)
#         sum2 = jnp.sum(outer_prod)
#         sum3 = (omega_i / (disp_i + mu_i) * (dg_y_exps + mu_i / (disp_i + mu_i)) * z_l**2).sum()
#         return (1 / 2) * (sum1 + sum2 - sum3)

#     mu_arr = intermed.compute_delta(model, inv = False)
#     disp_arr = model.disp_arr
#     omega_arr = intermed.compute_omega(model)
#     omega_delta_inv_arr = intermed.compute_delta_inv_omega(model)
#     z_l = model.data.z_arr[:, l]
#     kappa3 = model.compute_kappa(order = 3)
#     kappa4 = model.compute_kappa(order = 4)
#     values = np.stack([info_var_comp_l_disp_i(mu_i, disp_i, omega_i, omega_delta_inv_i, kappa3_i, kappa4_i, z_l) for mu_i, disp_i, omega_i, omega_delta_inv_i, kappa3_i, kappa4_i in zip(mu_arr, disp_arr, omega_arr, omega_delta_inv_arr, kappa3, kappa4)], axis = 0)
#     return jnp.expand_dims(jnp.array(values), 1)


# def info_var_comp_nuisance(model) -> jnp.array:
#     r"""Wrapper to compute the covariance of the variance component(s) and the nuisance parameters.

#     ARGS:
#         model - instance of Model class
#     RETURNS:
#         jax.numpy array; (n_targs, p + 1, q)
#     RAISES:
#         NONE
#     """
#     p = model.data.x_arr.shape[-1]
#     q = model.data.z_arr.shape[-1]
#     info_nuisance = jnp.zeros((model.data.n_targs, p + 1, q))

#     ## loop over variance components
#     for i in range(q):
#         w_fx_effs = info_var_comp_j_fx_effs(model, j = i)
#         w_disp    = info_var_comp_l_disp(model, l = i)
#         info_nuisance = info_nuisance.at[:, :-1, i].set(w_fx_effs)
#         info_nuisance = info_nuisance.at[:, -1:, i].set(w_disp)
#     return info_nuisance



# def info_eff_score(
#         model,
#         scale_val: float = None
#     ) -> jnp.array:
#     """
#     Compute the (scaled) information matrix of (U_eff, U_nuisance) under H0. 

#     ARGS:
#         model       - an instance of class Model
#         scale_val   - float; scaling value used to scale the score vector
#     RETURNS:
#          the inverse lower right block corresponding to the variance components covariances (q, q)
#     RAISES: None
#     """
#     q = model.data.z_arr.shape[-1]

#     ## (n_targs, q, q)
#     info_var = info_var_comp_var_comp(model)
#     ## (n_targs, p + 1, p + 1)
#     info_nuisance = info_nuisance_nuisance(model)
#     ## (n_targs, p + 1, q)
#     info_vc_n = info_var_comp_nuisance(model)
#     eff_info = np.zeros(q, q)
#     ## loop over targets
#     for i in range(info_nuisance.shape[0]):
#         eff_info += info_var[i, :, :] - info_vc_n[i, :, :].T @ jnp.linalg.pinv(info_nuisance[i, :, :]) @ info_vc_n[i, :, :]

#     if scale_val is not None:
#         return eff_info / scale_val**2
#     return eff_info

