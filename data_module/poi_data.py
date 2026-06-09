"""
Child class of `simul_data.Data` for Poisson outcomes.
"""

import numpy           as np
import jax.numpy       as jnp

from data_module import simul_data, helpers

class Data(simul_data.Data):
    def __init__(self, config):
        ## initialize parent class
        super().__init__(config)

    ## ---------------------------------------------------------------------- ##
    ## CHILD CLASS FUNCTIONS                                                  ##
    ## ---------------------------------------------------------------------- ##
    def set_dist_attrs(self):
        """
        Set Poisson distribution attributes.
        
        ARGS:    None
        RAISES:  None
        RETURNS: None
        """
        ## set column sums
        self.col_sums = self.draw_column_sums()
        # compute linear predictor
        self.eta_arr = self.compute_eta()
        ## set means
        self.mu_arr = jnp.exp(self.eta_arr)
        ## need to set disp in case we need to 'use_true' in neg_bin_model
        self.disp_arr = jnp.ones((self.n_targs, 1)) * np.inf

    def sample(self):
        """
        Sample responses as negative binomial. 

        ARGS:
            self
        RETURNS:
            y_arr - jnp.array; response array (n_targs, n_samps)
        RAISES:
            None
        """
        rng   = np.random.default_rng(None)
        y_arr = jnp.array(
            rng.poisson(lam = self.mu_arr)
        )
        return y_arr
