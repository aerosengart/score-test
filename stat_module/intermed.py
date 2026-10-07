"""Computations related to intermediate quantities."""

import jax
import jax.numpy as jnp
import numpy as np

def compute_delta(
        model,
        inv: bool = True
    ) -> jnp.array:
    """
    Compute and return the delta_i values.

    ARGS: model - an instance of class Model
    RETURNS: jnp.array (n_targs, n_samps)
    RAISES: None
    """
    ## compute delta_i values
    if inv:
        return model.compute_g_prime()
    else:
        return 1 / model.compute_g_prime()


def compute_omega(model) -> jnp.array:
    """
    Compute and return the omega_i values.

    L'Hopital:

    1 / (v * g'^2) = (1 / v) / g'^2

    derivative of top: - v' / v^2
    derivative of bottom: 2 g' * g''

    deriv of top = (1 + 2 * mu / disp) / (mu + mu^2 / disp)^2 
    deriv of bot = 2 * (1 / mu) *  (-1 / mu^2)

    ((1 + 2 * mu / disp) / (mu + mu^2 / disp)^2 ) / (2 * (1 / mu) *  (-1 / mu^2))
    = ((1 + 2 * mu / disp) / (mu * (1 + mu / disp))^2) / (-2 / mu^3)
    = - (mu^3 * (1 + 2 * mu / disp) / (mu^2 * (1 + mu / disp)^2 )) / 2
    = - mu * (1 + 2 * mu / disp) / (2 * (1 + mu / disp)^2)

    as mu -> 0, omega -> 0

    ARGS: model - an instance of class Model
    RETURNS: jnp.array (n_targs, n_samps)
    RAISES: None
    """
    # ## initial computations
    # g_prime_vals = model.compute_g_prime()
    # v_vals       = model.compute_v()
    # ## compute omega_i values
    # omegas = 1 / (v_vals * g_prime_vals**2)
    # nan_omega = np.where(np.isnan(omegas))
    # omegas = omegas.at[nan_omega].set(0.0)
    # return omegas
    mu_arr =  model.compute_mu()
    ## reparametrize with iota = 1 / gamma
    iota_arr = 1 / model.disp_arr * jnp.ones(model.data.n_samps)
    return mu_arr / (1 + iota_arr * mu_arr)

def compute_delta_inv_omega(model) -> jnp.array:
    """
    Compute omega_i * 1 / delta_i. 
    avoid NaNs
    (1 / delta) * (omega) = (1 / delta) * (delta^2 / v)
                          = delta / v
                          = (1 / g'(mu)) / (mu + mu^2 / disp)
                          = mu / (mu + mu^2 / disp)
                          = 1 / (1 + mu / disp) -> 1 as mu -> 0
    """
    omega_vals     = compute_omega(model)
    delta_inv_vals = compute_delta(model, inv = True)
    delta_inv_omega_vals = omega_vals * delta_inv_vals
    nan_vals = np.where(np.isnan(delta_inv_omega_vals))
    delta_inv_omega_vals = delta_inv_omega_vals.at[nan_vals].set(1.0)
    return delta_inv_omega_vals


def compute_epsilon(model) -> jnp.array:
    """
    Compute and return the epsilon_i values.

    Should equal zero for canonical links (Poisson case)

    numer = (1 + 2 * mu / disp) * (1 / mu) + (mu + mu^2 / disp) * (-1 / mu^2)
          = (1 / mu + 2 / disp) - (1 / mu + 1 / disp)
          = 1 / disp
    denom = (mu + mu^2 / disp)^2 * (1 / mu^3)
          = (mu * (1 + mu / disp))^2 / mu^3
          = (1 + mu / disp)^2 / mu

    as mu -> 0, numer -> 0 and denom -> Inf so eps -> 0

    ARGS: model - an instance of class Model
    RETURNS: jnp.array (n_targs, n_samps)
    RAISES: None
    """
    ## initial computations
    g_prime_vals   = model.compute_g_prime()
    g_prime_2_vals = model.compute_g_prime_2()
    v_vals         = model.compute_v()
    v_prime_vals   = model.compute_v_prime()
    ## compute epsilon_i values
    numer_vals = v_prime_vals * g_prime_vals + v_vals * g_prime_2_vals
    denom_vals = v_vals**2 * g_prime_vals**3
    eps_vals = numer_vals / denom_vals
    nan_eps = np.where(np.isnan(eps_vals))
    eps_vals = eps_vals.at[nan_eps].set(0.0)
    return eps_vals


def compute_xi(model) -> jnp.array:
    """
    Compute and return the xi_i values.

    Should not have any NaNs if no NaNs in omega or epsilon.

    ARGS: model - an instance of class Model
    RETURNS: jnp.array (n_targs, n_samps)
    RAISES: None
    """
    ## initial computations
    omega_vals   = compute_omega(model)
    epsilon_vals = compute_epsilon(model)
    resid_vals   = model.compute_resid()
    ## compute xi_i values
    return omega_vals + epsilon_vals * resid_vals

def compute_c(model) -> jnp.array:
    """
    Compute and return the c_i values.

    Should equal (omega_i * delta_i^{-1})**3 * kappa_{3,i} for canonical links (Poisson case).

    ARGS: model - an instance of class Model
    RETURNS: jnp.array (n_targs, n_samps)
    RAISES: None
    """
    ## initial computations
    epsilon_vals = compute_epsilon(model)
    kappa_2_vals = model.compute_kappa(order = 2)
    kappa_3_vals = model.compute_kappa(order = 3)
    delta_inv_omega_vals = compute_delta_inv_omega(model)

    ## compute c_i values
    term1 = delta_inv_omega_vals**3 * kappa_3_vals
    term2 = delta_inv_omega_vals * epsilon_vals * kappa_2_vals
    return term1 - term2

def compute_r(model) -> jnp.array:
    """
    Compute and return the R matrices for each cluster.

    ARGS: model - an instance of class Model
    RETURNS: jnp.array (n_targs, n_samps, n_samps)
    RAISES: None
    """
    ## initial computations
    omega_vals     = compute_omega(model)
    delta_inv_vals = compute_delta(model, inv = True)
    epsilon_vals   = compute_epsilon(model)
    kappa_2_vals = model.compute_kappa(order = 2)
    kappa_3_vals = model.compute_kappa(order = 3)
    kappa_4_vals = model.compute_kappa(order = 4)
    delta_inv_omega_vals = compute_delta_inv_omega(model)

    ## vectorize wrapper
    def wrap_func(omega_i, delta_inv_omega_i, epsilon_i, kappa_2_i, kappa_3_i, kappa_4_i):
        ## off-diagonal elements
        omega_omega = 2 * jnp.outer(omega_i, omega_i)

        ## diagonal elements
        term1 = delta_inv_omega_i**4 * kappa_4_i
        term2 = 2 * omega_i**2
        term3 = epsilon_i**2 * kappa_2_i
        term4 = 2 * delta_inv_omega_i**2 * epsilon_i * kappa_3_i
        diag_vals = term1 + term2 + term3 - term4

        r_arr = jnp.fill_diagonal(omega_omega, diag_vals, inplace = False)
        return r_arr
    ## compute values
    vmap_func = jax.vmap(wrap_func, in_axes = [0, 0, 0, 0, 0, 0])
    vals = vmap_func(
        omega_vals, delta_inv_omega_vals, epsilon_vals, 
        kappa_2_vals, kappa_3_vals, kappa_4_vals
    )
    return vals


def compute_a_j(
        model,
        j:   int = 0
    ) -> jnp.array:
    """
    Compute and return the A matrices for the j-th component of var_comp under the null hypothesis for all clusters.
    
    ARGS:
        model - an instance of class Model
        j     - int; index of variance component for derivative
    RETURNS: jnp.array (n_samps, n_samps)
    RAISES: None
    """
    D_var_comp_j = random_covar_wrt_var_comp_j(
        model,
        j = j
    )
    #z_j = jnp.expand_dims(model.data.z_arr[:, j], 1)
    z_arr = model.data.z_arr
    #return jnp.matmul(jnp.matmul(z_j, D_var_comp_j), jnp.transpose(z_j))
    return jnp.matmul(jnp.matmul(z_arr, D_var_comp_j), jnp.transpose(z_arr))

def random_covar_wrt_var_comp_j(
        model,
        j: int
    ) -> jnp.array:
    """
    Compute and return the matrix of first order partial derivatives of the random effects covariance matrix for a single cluster with respect to the j-th component.
    
    ARGS:
        model - an instance of class Model
        j     - int; index of random effects variance component vector with respect to which to take the partial derivatives
    RETURNS:
        jnp.array; first order partial derivatives of covariance matrix (q x q)
    """
    q = model.data.z_arr.shape[-1]
    q_zeros = jnp.zeros((q, q))
    covar_deriv = q_zeros.at[j, j].set(1)
    return covar_deriv

