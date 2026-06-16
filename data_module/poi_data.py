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
    ## COMPUTATIONS                                                           ##
    ## ---------------------------------------------------------------------- ##
    def compute_mu(self):
        """Compute and return linear predictor."""
        return jnp.exp(self.compute_eta())

    ## ---------------------------------------------------------------------- ##
    ## SAMPLERS                                                               ##
    ## ---------------------------------------------------------------------- ##
    def draw_disp(self):
        """Sample negative binomial dispersion parameter."""
        return jnp.ones((self.n_targs, 1)) * np.inf
    
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
        y_arr = jnp.array(
            self.rng.poisson(lam = self.compute_mu())
        )
        return y_arr
