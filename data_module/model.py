"""
Parent class for `model` object representing generalized linear mixed model for variance component score test according to Lin (1997). 
"""

import jax.numpy   as jnp

class Model():
    def __init__(self, config, data):
        """Constructor for class Model. This class organizes all of the attributes related to the model/approximation for the GLMM.

        ARGS:
            config - dict; configuration for class Model
            data   - an instance of class Data; the simulated data with which to fit the model
        RETURNS:
            None
        RAISES:
            None
        """
        ## for doc.py (should remove this and edit that file to read attrbs)
        self.config = config
        
        ## save data
        self.data = data
        ## model housekeeping
        self.outcome      = config['outcome']
        self.scale        = config['scale']
        self.use_true     = config['use_true']
        self.debug        = config['debug']
        self.size_factors = config['size_factors']
        ## set whether dispersion is allowed to be target-specific`
        if config['global_disp']:
            self.estim_disp = 'global'
        else:
            self.estim_disp = True

        ## !! these are not the MLEs --- they are the values under H0 !!
        self.set_rd_effs_arr(jnp.zeros((self.data.n_targs, self.data.q)))
        self.set_tau_sqr(jnp.zeros((self.data.q)))
        ## estimate the other parameters via maximum likelihood
        self.set_dist_params_estim()

    ## ---------------------------------------------------------------------- ##
    ## GETTERS/SETTERS                                                        ##
    ## ---------------------------------------------------------------------- ##
    def set_fx_effs_arr(self, fx_effs_arr):
        self.fx_effs_arr = fx_effs_arr

    def set_rd_effs_arr(self, rd_effs_arr):
        self.rd_effs_arr = rd_effs_arr

    def set_tau_sqr(self, tau_sqr):
        self.tau_sqr = tau_sqr

    def set_mu_arr(self, mu_arr):
        self.mu_arr = mu_arr

    ## ---------------------------------------------------------------------- ##
    ## INTERMEDIATE QUANTITIES                                                ##
    ## ---------------------------------------------------------------------- ##
    def compute_kappa(
            self,
            order: int = 2
        ) -> jnp.array:
        """
        Compute and return the kappa terms of provided order for cluster i. For a single observation, kappa of order j is the observation's j-th cumulant.

        Assumes order k cumulants are assumed to be equal to second cumulant times derivative of order k - 1 cumulant.

        ARGS:
            fx_effs_arr - jnp.array; optional array of estimated fixed effects (n, k, p)
            order       - int; order of the cumulant
        RETURNS:
            jnp.array; array of cumulants (n, k, 1)
        RAISES:
            NotImplementedError - if order > 4
        """
        ## second cumulant is variance
        kappa_2 = self.compute_v()
        if order == 2:
            return kappa_2
        elif order == 3:
            deriv_kappa_2 = self.compute_v_prime()
            return kappa_2 * deriv_kappa_2
        elif order == 4:
            deriv_var     = self.compute_v_prime()
            deriv_2_var   = self.compute_v_prime_2()
            deriv_kappa_3 = kappa_2 * deriv_2_var + deriv_var**2
            return kappa_2 * deriv_kappa_3
        else:
            raise NotImplementedError("Order must be less than or equal to 4.")

    ## ---------------------------------------------------------------------- ##
    ## CHILD CLASS FUNCTIONS                                                  ##
    ## ---------------------------------------------------------------------- ##
    def set_disp_param(self, disp_param):
        raise NotImplementedError("Subclasses must implement this method.")

    def set_dist_param(self, disp_param):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def set_dist_params_estim(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def compute_resid(self) -> jnp.array:
        raise NotImplementedError("Subclasses must implement this method.")

    def compute_g(self) -> jnp.array:
        raise NotImplementedError("Subclasses must implement this method.")

    def compute_g_prime(self) -> jnp.array:
        raise NotImplementedError("Subclasses must implement this method.")

    def compute_g_prime_2(self) -> jnp.array:
        raise NotImplementedError("Subclasses must implement this method.")

    def compute_v(self) -> jnp.array:
        raise NotImplementedError("Subclasses must implement this method.")

    def compute_v_prime(self) -> jnp.array:
        raise NotImplementedError("Subclasses must implement this method.")
    
    def compute_v_prime_2(self) -> jnp.array:
        raise NotImplementedError("Subclasses must implement this method.")
    
    def compute_dispersion(self):
        raise NotImplementedError("Subclasses must implement this method.")


