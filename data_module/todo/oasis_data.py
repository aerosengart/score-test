"""
Child class of `simul_data.Data` for OASIS-type (multinomial) data.
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
        Set negative binomial distribution attributes.
        
        ARGS:    None
        RAISES:  None
        RETURNS: None
        """
        ## set gamma
        self.set_gamma(self.draw_gamma())
        ## set column sums
        self.col_sums = self.draw_column_sums()
        # compute linear predictor
        self.eta_arr = self.compute_eta() + jnp.log(self.col_sums)
        ## set means
        self.mu_arr = jnp.exp(self.eta_arr)
        ## set variances
        self.var_arr = self.mu_arr + jnp.divide(self.mu_arr**2, self.gamma)
        ## compute success probability for (r, p) parametrization in numpy
        ## r = # successes; p = success probability
        ## p = mean / var = mu / (mu + mu^2/gamma) = gamma / (gamma + mu)
        ## r = mean^2 / (var - mean) = mu^2 / (mu^2/gamma) = gamma
        self.probs = jnp.divide(self.gamma, jnp.add(self.mu_arr, self.gamma))
        self.succs = self.gamma

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
            rng.negative_binomial(n = self.succs, p = self.probs)
        )
        ## quick fix to avoid rows of only zeroes
        while 0 in jnp.sum(y_arr, axis = 1):
            y_arr = jnp.array(
                rng.negative_binomial(n = self.succs, p = self.probs)
            )
        return y_arr

    def draw_gamma(self):
        """Sample negative binomial dispersion parameter."""
        gamma = helpers.draw_gaussian_arr(
            row   = self.n_targs,
            col   = 1,
            mu    = self.disp_mu,
            sigma = self.disp_sigma
        )
        gamma = gamma.at[gamma <= 0.0].set(1e-4) ## avoid zero/negative gamma
        return gamma



    def set_gamma(self, gamma):
        """Setter for negative binomial dispersion parameter."""
        self.gamma = gamma
    
