"""
Child class of `simul_data.Data` for negative binomial outcomes.
"""

import numpy           as np
import jax.numpy       as jnp

from data_module import simul_data

class Data(simul_data.Data):
    def __init__(self, config):
        ## initialize parent class
        super().__init__(config)

        ## number of samples
        self.n_samps     = int(config['n_samps'])
        ## number of targets
        self.n_targs     = int(config['n_targs'])

        ## mean and standard deviation for fixed effects
        self.fx_effs_mu      = float(config['fx_effs']['mu'])
        self.fx_effs_sigma   = float(config['fx_effs']['sigma'])
        ## mean and standard deviation for fixed effects covariates
        self.x_mu      = float(config['x']['mu'])
        self.x_sigma   = float(config['x']['sigma'])
        ## mean and standard deviation for random effects covariates
        self.z_mu    = float(config['z']['mu'])
        self.z_sigma = float(config['z']['sigma'])
        ## set tau_sqr
        self.tau_sqr = jnp.array(config['tau_sqr'])

        ## set total number of fixed effects (accounting for target-specific)
        self.p_star = self.n_targs + self.fx_slopes * self.n_targs

        ## set covariates: (n_samps, p), (n_samps, q)
        x_arr, z_arr = self.get_covars()
        self.set_covars(x_arr, z_arr)

        ## set coefficients: (n_targs, p), (n_targs, q)
        fx_effs_arr, rd_effs_arr = self.draw_effs()
        self.set_coefs(fx_effs_arr, rd_effs_arr)

        # compute linear predictor
        self.eta_arr = self.compute_eta()

        ## draw responses
        self.set_dist_attrs(disp_param = config['disp_param'])
        targ_idx, samp_idx, counts = self.get_tab_vals()
        self.set_tab_vals(targ_idx, samp_idx, counts)

        ## debugging dictionary
        self.debug = {
            'iter':    'default',
            'rep':     'default',
            'routine': 'default'
        }

    ## ---------------------------------------------------------------------- ##
    ## CHILD CLASS FUNCTIONS                                                  ##
    ## ---------------------------------------------------------------------- ##
    def sample(self):
        """
        Sample responses as negative binomial. 

        ARGS:
            self
        RETURNS:
            y_arr - jnp.array; response array (n, k, 1)
        RAISES:
            None
        """
        rng   = np.random.default_rng(None)
        y_arr = jnp.array(
            rng.negative_binomial(n = self.succs, p = self.probs)
        )
        ## quick fix to avoid rows of only zeroes
        while 0 in jnp.sum(y_arr, axis = 1):
            y_arr = jnp.array(
                rng.negative_binomial(n = self.succs, p = self.probs)
            )
        return y_arr

    def set_dist_attrs(self, disp_param):
        """
        Set negative binomial distribution attributes.
        
        ARGS:    None
        RAISES:  None
        RETURNS: None
        """        
        ## set gamma
        self.set_disp_param(disp_param)
        ## set means
        self.mu_arr = jnp.exp(self.eta_arr)
        ## vars
        self.var_arr = self.mu_arr + jnp.divide(self.mu_arr**2, self.gamma)
        ## compute success probability for (r, p) parametrization in numpy
        ## r = # successes; p = success probability
        ## r = gamma; p = r / (m + r)
        self.probs = jnp.divide(self.gamma, jnp.add(self.mu_arr, self.gamma))
        self.succs = self.gamma

    def set_disp_param(self, disp_param):
        """
        Set negative binomial dispersion/scale parameter.
        """
        self.gamma = disp_param







    
    # def load_from_file(self, path):
    #     """Read data from .npz file."""
    #     d = np.load(path)
    #     ## extract from dictionary
    #     self.y_arr       = d['y_arr']
    #     self.x_arr       = d['x_arr']
    #     self.z_arr       = d['z_arr']
    #     self.fx_effs_arr = d['fx_effs_arr']
    #     self.rd_effs_arr = d['rd_effs_arr']
    #     self.mu_arr      = d['mu_arr']
    #     self.var_arr     = self.mu_arr + jnp.divide(self.mu_arr**2, self.gamma)
    #     self.probs_arr   = d['probs_arr']

