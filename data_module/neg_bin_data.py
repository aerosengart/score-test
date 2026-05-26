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
        
        ARGS: None
        RAISES: None
        RETURNS: None
        """        
        ## set gamma
        self.gamma = jnp.array(disp_param)
        ## set means
        self.mu_arr = jnp.exp(self.eta_arr)
        ## vars
        self.var_arr = self.mu_arr + jnp.divide(self.mu_arr**2, self.gamma)
        ## compute success probability for (r, p) parametrization in numpy
        ## r = # successes; p = success probability
        ## r = gamma; p = r / (m + r)
        self.probs = jnp.divide(self.gamma, jnp.add(self.mu_arr, self.gamma))
        self.succs = self.gamma

    
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

