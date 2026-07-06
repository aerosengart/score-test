"""
Child class of `simul_data.Data` for approximately multinomial outcomes (via a Poisson approximation).
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
        ## set means
        temp_mu_arr = jnp.exp(self.compute_eta())
        ## baseline probability for targets --- column sums over all rows but first
        col_sums_m1 = jnp.sum(temp_mu_arr, axis = 0) - temp_mu_arr[0, :]
        ## baseline probabilities by sample (corresponds to prob of first target)
        p_bl = jnp.expand_dims(jnp.divide(1, 1 + col_sums_m1), 0)
        p_others = p_bl * temp_mu_arr[1:, :]
        self.probs = jnp.concatenate([p_bl, p_others], axis = 0)
        ## sample column sums
        self.col_sums = self.draw_column_sums()
        ## compute multinomial mean/variance
        self.mu_arr  = jnp.multiply(
            self.probs, 
            jnp.tile(self.col_sums, (self.n_targs, 1))
        )
        self.var_arr = jnp.multiply(
            self.probs, 
            jnp.ones(self.probs.shape) - self.probs
        )


    def sample(self):
        """
        Sample responses as multinomial

        ARGS:
            self
        RETURNS:
            y_arr - jnp.array; response array (n_targs, n_samps)
        RAISES:
            None
        """
        y_arr = jnp.array(
            self.rng.multinomial(n = self.col_sums, pvals = self.probs.T).T
        )
        ## ensure no row sums to zero (null target)
        while 0 in jnp.sum(y_arr, axis = 1):
            y_arr = jnp.array(
                self.rng.multinomial(n = self.col_sums, pvals = self.probs.T).T
            )
        return y_arr

