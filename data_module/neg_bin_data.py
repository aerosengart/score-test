"""
Child class of `simul_data.Data` for negative binomial outcomes.
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
        ## compute success probability for (r, p) parametrization in numpy
        ## r = # successes; p = success probability
        ## p = mean / var = mu / (mu + mu^2/disp) = disp / (disp + mu)
        ## r = mean^2 / (var - mean) = mu^2 / (mu^2/disp) = disp
        probs = jnp.divide(self.disp_arr, jnp.add(self.compute_mu(), self.disp_arr))
        succs = self.disp_arr
        y_arr = jnp.array(
            self.rng.negative_binomial(n = succs, p = probs)
        )
        ## quick fix to avoid rows of only zeroes
        while 0 in jnp.sum(y_arr, axis = 1):
            y_arr = jnp.array(
                self.rng.negative_binomial(n = succs, p = probs)
            )
        return y_arr
    

    def draw_disp(self):
        """Sample negative binomial dispersion parameter."""
        if self.disp_mu != float('inf'):
            disp_arr = helpers.draw_gaussian_arr(
                row   = self.n_targs,
                col   = 1,
                mu    = self.disp_mu,
                sigma = self.disp_sigma,
                rng   = self.rng,
            )
        else:
            disp_arr = self.disp_mu * jnp.ones((self.n_targs, 1))
        disp_arr = disp_arr.at[disp_arr < 0.0].set(1e-4) ## avoid negative disp
        return disp_arr

