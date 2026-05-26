"""
Subclass of `Model` object for negative binomial-like variance function. The overdispersion parameter (phi) for the quasi-likelihood is assumed to be fixed at 1. Random effects are assumed to be independent (diagonal covariance).

The likelihood functions are based upon the negative binomial distribution, which is NOT the same as the quasi-likelihood methods used in the test derivations.

NEGATIVE BINOMIAL ATTRIBUTES:
    gamma_mle       - jnp.array; negative binomial dispersion/scale parameter (n, k, 1) --- permitted to be cluster-specific 
"""

import jax.numpy       as jnp
import scipy.sparse    as sps

from data_module  import model
from score_module import glm_fit

class Model(model.Model):
    def __init__(self, config, data):
        ## initialize parent class
        super().__init__(config, data)

    ## ---------------------------------------------------------------------- ##
    ## GETTERS/SETTERS                                                        ##
    ## ---------------------------------------------------------------------- ##
    def set_gamma(self, gamma):
        """Set negative binomial dispersion/scale parameter."""
        self.gamma = gamma
        
    def set_dist_params_estim(self):
        """Compute the maximum likelihood estimate for the fixed effects and the negative binomial dispersion/scale parameters.

        ARGS:
            self
        RETURNS:
            None
        RAISES:
            None
        """
        ## use glmGamPoi if we have "real" data (i.e null model is a cluster-specific intercept)
        fx_effs, disp = glm_fit.glmgampoi_fit(model = self)
        self.set_fx_effs_arr(fx_effs)
        self.set_gamma(disp)


    ## ---------------------------------------------------------------------- ##
    ## COMPUTATIONS                                                           ##
    ## ---------------------------------------------------------------------- ##
    def compute_eta(self) -> jnp.array:
        """
        Compute linear predictor array with parameter estimates/values under H0. 

        ARGS:    None
        RAISES:  None
        RETURNS: jnp.array (n_samps)
        """
        ## linear predictor
        fx_part = self.fx_effs_arr @ self.data.x_arr.T
        rd_part = self.rd_effs_arr @ self.data.z_arr.T
        return fx_part + rd_part


    def compute_mu(self) -> jnp.array:
        """
        Compute mean array with parameter estimates/values under H0. 

        ARGS:    None
        RAISES:  None
        RETURNS: jnp.array (n_targs, n_samps)
        """
        return jnp.exp(self.compute_eta())


    def compute_resid(self) -> jnp.array:
        """
        Compute and return residual vector (y - mu) under H0.

        ARGS: None
        RAISES: None
        RETURNS: jnp.array (n_targs, n_samps)
        """
        ## (n_targs, n_samps)
        exp_val = self.compute_mu()
        exp_val_csc = sps.csc_array(exp_val)
        y_arr = self.data.construct_y()
        resid_arr = y_arr - exp_val_csc
        return jnp.array(resid_arr.todense())


    def compute_g(self) -> jnp.array:
        """
        Compute and return the linear predictor values under H0.
        
        Computes g(mu) = eta.

        ARGS:   None
        RAISES: None
        RETURNS: jnp.array (n_targs, n_samps)
        """
        return self.compute_eta()


    def compute_g_prime(self) -> jnp.array:
        r"""
        Compute and return the array of first derivatives of linear predictor (:math:`g'(\mu_i)`) with respect to mean under H0.

        For the negative binomial case, this is 1 / mu_i.

        ARGS:   None
        RAISES: None
        RETURNS: jnp.array (n_targs, n_samps)
        """
        return 1 / self.compute_mu()


    def compute_g_prime_2(self) -> jnp.array:
        r"""
        Compute and return the diagonal array of second derivatives of linear predictor (:math:`g''(\mu_i)`) on main diagonal.

        For the negative binomial, this is -1 / mu_i^2.

        ARGS:   None
        RAISES: None
        RETURNS: jnp.array (n_targs, n_samps)
        """
        return -1 / self.compute_mu()**2


    def compute_v(self) -> jnp.array:
        """
        Compute and return the diagonal array by concatenating the cluster-specific vectors of variances.

        For the negative binomial case, this is mu + mu^2 / gamma.

        ARGS: None
        RAISES: None
        RETURNS: jnp.array; diagonal array (n, k, 1)
        """
        mu_arr = self.compute_mu()
        prod = (1 / self.gamma) * mu_arr**2
        return mu_arr + prod


    def compute_v_prime(self) -> jnp.array:
        """
        Compute and return the diagonal array by concatenating the cluster-specific vectors of first derivatives of variances with respect to the means.

        For the negative binomial case, this is 1 + 2 * mu / gamma.

        ARGS: None
        RAISES: None
        RETURNS: jnp.array; diagonal array (n, k, 1)
        """        
        mu_arr = self.compute_mu()
        prod = (2 / self.gamma) * mu_arr
        return 1 + prod

    def compute_v_prime_2(self) -> jnp.array:
        """
        Compute and return the diagonal array by concatenating the cluster-specific vectors of second derivatives of variances with respect to the means under H0.

        For the negative binomial case, this is 2 / gamma.

        ARGS: None
        RAISES: None
        RETURNS: jnp.array; diagonal array (n, k, 1)
        """        
        return 2 / self.gamma

    
