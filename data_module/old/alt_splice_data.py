"""
Child class of `simul_data.Data` for Poisson outcomes in alternative splicing-type setting.

NOTE: There is NO GROUND TRUTH for our method because the pure signal setting is not possible with our assumed data-generating process.
"""


import math
import numpy           as np
import jax.numpy       as jnp

from data_module import simul_data, helpers

class Data(simul_data.Data):
    def __init__(self, config):
        ## expand on config to fit Data implementation
        config['x'] = {}
        config['z'] = {}
        config['x']['types'] = []
        config['z']['types'] = ['batch_2']

        ## set corruption value
        self.set_corrupt(float(config['corruption']))

        ## initialize parent class
        super().__init__(config)

        self.resample()

    def set_corrupt(self, corrupt):
        self.corrupt = corrupt

    ## ---------------------------------------------------------------------- ##
    ## COMPUTATIONS                                                           ##
    ## ---------------------------------------------------------------------- ##
    def compute_eta(self):
        """
        Compute the "linear predictor". This is effectively the linear predictor but is actually the row success probabilities for the multinomial samples (columns) of the count table.
        """
        eta = np.ones((self.n_targs, self.n_samps)) * (self.corrupt / self.n_targs)
        eta[0, :math.floor(self.n_samps / 2)] += 1 - self.corrupt
        eta[1, math.floor(self.n_samps / 2):] += 1 - self.corrupt
        return eta
    
    def compute_mu(self):
        """Compute the observation means. This is the mean in OUR GLMM setting NOT the true multinomial means (this requires multiplying by column sum). We need the means in our setting to do the computations for our test, and we never need the true multinomial means (this is computed in OASIS separately)."""
        return jnp.exp(self.compute_eta())

    ## ---------------------------------------------------------------------- ##
    ## SAMPLERS                                                               ##
    ## ---------------------------------------------------------------------- ##
    def draw_effs(self):
        """
        "Draw" fixed and random effects. These are just None for this scenario.
        """
        return None, None

    def draw_disp(self):
        """Sample negative binomial dispersion parameter."""
        return jnp.ones((self.n_targs, 1)) * np.inf

    def draw_col_sums(self):
        """
        Draw column sums according to Fig. S6 in OASIS supplement.
        """
        col_sums = jnp.array(
            self.rng.poisson(lam = 20, size = (self.n_samps,))
        )
        return col_sums, jnp.zeros((self.n_samps, ))

    def draw_covars(self):
        """
        Draw covariate arrays. We assume to have only the target-specific fixed intercept and a grouping factor for the random slopes. We assume to KNOW the grouping that dictates the signal as in Fig. S6 of OASIS supplement.
        """
        ## equal sized groups
        x_arr = jnp.ones((self.n_samps, 1))
        ## equal sized groups
        z_arr = jnp.ones((self.n_samps, 1))
        z_arr = z_arr.at[:math.floor(self.n_samps / 2), :].set(-1.0)
        return x_arr, z_arr
        
    def sample(self):
        """
        Sample responses as multinomial.

        ARGS:    None
        RETURNS: y_arr - jnp.array; response array (n_targs, n_samps)
        RAISES:  None
        """
        sample = self.rng.multinomial(n = self.col_sums, pvals = self.compute_eta().T)
        y_arr = jnp.array(sample.T)
        return y_arr
