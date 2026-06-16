"""Computations related to intermediate quantities."""

import jax
import jax.numpy as jnp

def compute_c(model) -> jnp.array:
    """
    Compute and return the c_i values.

    Should equal (omega_i * delta_i^{-1})**3 * kappa_{3,i} for canonical links (Poisson case).

    ARGS: model - an instance of class Model
    RETURNS: jnp.array (n_targs, n_samps)
    RAISES: None
    """
    ## initial computations
    delta_inv_omega_vals = model.compute_delta_inv_omega()
    epsilon_vals = model.compute_epsilon()
    kappa_2_vals = model.compute_kappa(order = 2)
    kappa_3_vals = model.compute_kappa(order = 3)
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
    omega_vals           = model.compute_omega()
    delta_inv_omega_vals = model.compute_delta_inv_omega()
    epsilon_vals         = model.compute_epsilon()
    kappa_2_vals = model.compute_kappa(order = 2)
    kappa_3_vals = model.compute_kappa(order = 3)
    kappa_4_vals = model.compute_kappa(order = 4)

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
    z_j = jnp.expand_dims(model.data.z_arr[:, j], 1)
    return jnp.matmul(jnp.matmul(z_j, D_var_comp_j), jnp.transpose(z_j))

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
    q_zeros = jnp.zeros((model.data.q, model.data.q))
    covar_deriv = q_zeros.at[j, j].set(1)
    return covar_deriv

