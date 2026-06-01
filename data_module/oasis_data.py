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
        mn_arr = jnp.array(
            rng.multinomial(n = self.col_sums, pvals = self.probs).T
        )
        return mn_arr

    def set_dist_attrs(self, disp_param):
        """
        Set negative binomial distribution attributes.
        
        ARGS: None
        RAISES: None
        RETURNS: None
        """
        ## set means
        self.mu_arr = jnp.exp(self.eta_arr)
        ## calculate row probabilities --- E[Y_ij] = n_j * p_i
        ## assume all columns have sum = 100
        col_sum = 100
        p_vec = (self.mu_arr[:, 0] / col_sum) / jnp.sum(self.mu_arr[:, 0] / col_sum)
        self.probs = p_vec
        self.col_sums = [col_sum] * self.n_samps
        





