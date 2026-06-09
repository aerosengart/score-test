"""Child class of `Model` for negative binomial responses."""

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
    def set_disp_arr(self, disp_arr):
        """Set negative binomial dispersion/scale parameter."""
        self.disp_arr = disp_arr
        
    def set_dist_params_estim(self):
        """Compute the maximum likelihood estimate for the fixed effects and the negative binomial dispersion/scale parameters. Also compute the estimated means because there are adjustments made by glmGamPoi.

        ARGS:
            self
        RETURNS:
            None
        RAISES:
            None
        """
        if self.use_true:
            fx_effs  = self.data.fx_effs_arr
            disp_arr = self.data.disp_arr
            mu_arr   = self.data.mu_arr
            offsets  = self.data.offsets
        else:
            fx_effs, disp_arr, offsets, mu_arr = glm_fit.glmgampoi_fit(model = self)
        self.set_fx_effs_arr(fx_effs)
        self.set_disp_arr(disp_arr)
        self.set_offsets(offsets)
        self.set_mu_arr(mu_arr)

    def update_params(self):
        """
        Update model attributes downstream of fx_effs_arr, eta_arr, mu_arr, disp_arr, or offsets. This includes the linear predictor array (eta_arr) and the mean array (mu_arr).

        ARGS:    None
        RAISES:  None
        RETURNS: None
        """
        self.set_eta_arr(self.compute_eta())
        self.set_mu_arr(self.compute_mu())

    ## ---------------------------------------------------------------------- ##
    ## COMPUTATIONS                                                           ##
    ## ---------------------------------------------------------------------- ##
    def compute_mu(self):
        """
        Compute mean array using class attributes.

        ARGS: None
        RAISES: None
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
        exp_val = self.mu_arr
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
        return jnp.exp(self.mu_arr) 

    def compute_g_prime(self) -> jnp.array:
        r"""
        Compute and return the array of first derivatives of linear predictor (:math:`g'(\mu_i)`) with respect to mean under H0.

        For the negative binomial case, this is 1 / mu_i.

        ARGS:   None
        RAISES: None
        RETURNS: jnp.array (n_targs, n_samps)
        """
        return 1 / self.mu_arr 


    def compute_g_prime_2(self) -> jnp.array:
        r"""
        Compute and return the diagonal array of second derivatives of linear predictor (:math:`g''(\mu_i)`) on main diagonal.

        For the negative binomial, this is -1 / mu_i^2.

        ARGS:   None
        RAISES: None
        RETURNS: jnp.array (n_targs, n_samps)
        """
        return -1 / self.mu_arr**2 
    

    def compute_v(self) -> jnp.array:
        """
        Compute and return the diagonal array by concatenating the cluster-specific vectors of variances.

        For the negative binomial case, this is mu + mu^2 / disp_arr.

        ARGS: None
        RAISES: None
        RETURNS: jnp.array; diagonal array (n, k, 1)
        """
        mu_arr = self.mu_arr
        prod = (1 / self.disp_arr) * mu_arr**2
        return mu_arr + prod


    def compute_v_prime(self) -> jnp.array:
        """
        Compute and return the diagonal array by concatenating the cluster-specific vectors of first derivatives of variances with respect to the means.

        For the negative binomial case, this is 1 + 2 * mu / disp_arr.

        ARGS: None
        RAISES: None
        RETURNS: jnp.array; diagonal array (n, k, 1)
        """        
        mu_arr = self.mu_arr
        prod = (2 / self.disp_arr) * mu_arr
        return 1 + prod


    def compute_v_prime_2(self) -> jnp.array:
        """
        Compute and return the diagonal array by concatenating the cluster-specific vectors of second derivatives of variances with respect to the means under H0.

        For the negative binomial case, this is 2 / disp_arr.

        ARGS: None
        RAISES: None
        RETURNS: jnp.array; diagonal array (n, k, 1)
        """        
        return 2 / self.disp_arr

    
