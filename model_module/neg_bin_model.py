"""Child class of `Model` for negative binomial responses."""

import jax.numpy       as jnp
import scipy.sparse    as sps
import numpy           as np

from model_module import model
from model_module import glmgampoi

class Model(model.Model):
    def __init__(self, config, data):
        ## initialize parent class
        super().__init__(config, data)

        ## set whether dispersion is allowed to be target-specific and/or fixed
        if isinstance(config['global_disp'], bool):
            if config['global_disp'] == True:
                self.estim_disp = 'global'
            else:
                self.estim_disp = True
        else:
            ## set dispersion to be some fixed value specific in config file
            ## reciprocal because of parametrization in glmGamPoi
            self.estim_disp = [1 / x for x in config['global_disp']]

        ## estimate the other parameters via maximum likelihood
        self.set_dist_params_estim()

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
            mu_arr   = self.data.compute_mu()
            offsets  = self.data.offsets
        else:
            fx_effs, disp_arr, offsets, mu_arr = glmgampoi.fit(model = self)
        self.set_fx_effs_arr(fx_effs)
        self.set_disp_arr(disp_arr)
        self.set_offsets(offsets)
        self.set_mu_arr(mu_arr)

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
        exp_val = self.compute_mu()
        exp_val_csc = sps.csc_array(exp_val)
        y_arr = self.data.construct_y()
        resid_arr = y_arr - exp_val_csc
        return jnp.array(resid_arr)

    def compute_g(self) -> jnp.array:
        """
        Compute and return the linear predictor values under H0.
        
        Computes g(mu) = eta = log(mu)

        ARGS:   None
        RAISES: None
        RETURNS: jnp.array (n_targs, n_samps)
        """
        return jnp.exp(self.compute_mu())

    def compute_g_prime(self) -> jnp.array:
        r"""
        Compute and return the array of first derivatives of linear predictor (:math:`g'(\mu_i)`) with respect to mean under H0.

        For the negative binomial case, this is 1 / mu_i.

        ARGS:   None
        RAISES: None
        RETURNS: jnp.array (n_targs, n_samps)
        """
        g_prime = 1 / self.compute_mu()
        zero_mu = np.where(self.compute_mu() == 0)
        g_prime = g_prime.at[zero_mu].set(np.inf)
        return g_prime


    def compute_g_prime_2(self) -> jnp.array:
        r"""
        Compute and return the diagonal array of second derivatives of linear predictor (:math:`g''(\mu_i)`) on main diagonal.

        For the negative binomial, this is -1 / mu_i^2.

        ARGS:   None
        RAISES: None
        RETURNS: jnp.array (n_targs, n_samps)
        """
        g_prime_2 = -1 / self.compute_mu()**2 
        zero_mu = np.where(self.compute_mu() == 0)
        g_prime_2 = g_prime_2.at[zero_mu].set(-np.inf)
        return g_prime_2
    

    def compute_v(self) -> jnp.array:
        """
        Compute and return the diagonal array by concatenating the cluster-specific vectors of variances.

        For the negative binomial case, this is mu + mu^2 / disp_arr.

        ARGS: None
        RAISES: None
        RETURNS: jnp.array; diagonal array (n, k, 1)
        """
        mu_arr = self.compute_mu()
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
        mu_arr = self.compute_mu()
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

    
    ## ---------------------------------------------------------------------- ##
    ## INTERMEDIATE QUANTITIES                                                ##
    ##     have to overwrite to avoid numerical issues (divide by zero)       ##
    ## ---------------------------------------------------------------------- ##
    def compute_delta_inv_omega(self) -> jnp.array:
        """
        Compute and return the values of delta_i^{-1} * omega_i for negative binomial outcomes. This is:

        delta^2 / v(mu) * 1 / delta = delta / v(mu)

        ARGS:    None
        RETURNS: jnp.array (n_targs, n_samps)
        RAISES:  None
        """
        delta_inv = self.compute_mu()
        variances = self.compute_v()
        return delta_inv / variances
    
    def compute_omega(self) -> jnp.array:
        """
        Compute and return the omega_i values for negative binomial responses. This is:

        delta^2 / v(mu) = mu^2 / (mu + mu^2 / disp) = mu / (1 + mu / disp)

        ARGS:    None
        RETURNS: jnp.array (n_targs, n_samps)
        RAISES:  None
        """
        return self.compute_mu() / (1 + self.compute_mu() / self.disp_arr)

    def compute_epsilon(self) -> jnp.array:
        """
        Compute and return the epsilon_i values for negative binomial response. 
        ARGS:    None
        RETURNS: jnp.array (n_targs, n_samps)
        RAISES:  None
        """
        ## initial computations
        g_prime_vals   = self.compute_g_prime()
        g_prime_2_vals = self.compute_g_prime_2()
        v_vals         = self.compute_v()
        v_prime_vals   = self.compute_v_prime()
        ## compute epsilon_i values
        numer_vals = v_prime_vals * g_prime_vals + v_vals * g_prime_2_vals
        denom_vals = v_vals**2 * g_prime_vals**3
        return numer_vals / denom_vals


    def compute_xi(self) -> jnp.array:        
        """
        Compute and return the xi_i values for negative binomial responses. This is:

        omega + eps * (y - mu)

        ARGS:    None
        RETURNS: jnp.array (n_targs, n_samps)
        RAISES:  None
        """
        return self.compute_omega() + self.compute_epsilon() * self.compute_resid()
