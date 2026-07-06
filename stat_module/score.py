"""Computations related to the score function."""

import math
import jax
import jax.numpy    as jnp
import numpy        as np
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
    xi_vals        = intermed.compute_xi(model)
    delta_inv_omega_vals = intermed.compute_delta_inv_omega(model)
    for idx in range(model.data.q):
        ## (n_samps, n_samps) --- same for all targets
        a_j_arr = intermed.compute_a_j(model, j = idx)

        ## vectorize wrapper --- compute target-by-target
        def wrap_func(resid_targ, delta_inv_omega_targ, xi_targ):
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
    
    if np.isnan(wrt_var_comp):
        doc.debug_sample(model)
        raise Exception
    return wrt_var_comp, scale_val

