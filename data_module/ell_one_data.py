"""
Child class of `simul_data.Data` for Poisson outcomes in l1 corruption scenario.

NOTE: There is NO GROUND TRUTH for our method because the pure signal setting is not possible with our assumed data-generating process.
"""


import math
import numpy           as np
import jax.numpy       as jnp

from data_module import scenario_data, helpers

class Data(scenario_data.Data):
    def __init__(self, config):
        ## initialize parent class
        super().__init__(config)

        ## things that we don't need to specify in the config file
        self.outcome = 'multinom'
        self.x_types = []
        self.z_types = ['time_reg']
        self.p = 1
        self.q = 1

        ## set corruption value
        self.set_corrupt(float(config['corruption']))

        ## draw mixture distribution (q_js in Eq. 43)
        self.mix_dist = self.draw_mix_dist()


    ## ---------------------------------------------------------------------- ##
    ## GETTERS / SETTERS                                                      ##
    ## ---------------------------------------------------------------------- ##
    def get_attrbs(self):
        """Construct data attributes dictionary."""
        dim_dict = {
            'n_targs':   self.n_targs,
            'n_samps':   self.n_samps,
            'rel_targs': self.rel_targs,
            'rel_samps': self.rel_samps
        }
        x_dict = {
            'center': self.ctr_x,
            'types': self.x_types
        }
        z_dict = {
            'center': self.ctr_z,
            'types': self.z_types
        }
        param_dict = {
            'outcome': self.outcome,
            'dims': dim_dict,
            'corrupt': self.corrupt,
            'x': x_dict,
            'z': z_dict,
        }
        return param_dict
    
    def set_corrupt(self, corrupt):
        self.corrupt = corrupt

    ## ---------------------------------------------------------------------- ##
    ## COMPUTATIONS                                                           ##
    ## ---------------------------------------------------------------------- ##
    def compute_eta(self):
        """
        Compute the "linear predictor". This is effectively the linear predictor but is actually the row success probabilities for the multinomial samples (columns) of the count table.
        """
        eta = np.ones((self.n_targs, self.n_samps)) * (1 - self.corrupt) / self.n_targs
        eta += self.corrupt * self.mix_dist
        return eta
    
    def compute_mu(self):
        """Compute the observation means. This is the mean in OUR GLMM setting NOT the true multinomial means (this requires multiplying by column sum). We need the means in our setting to do the computations for our test, and we never need the true multinomial means (this is computed in OASIS separately)."""
        return jnp.exp(self.compute_eta())

    ## ---------------------------------------------------------------------- ##
    ## SAMPLERS                                                               ##
    ## ---------------------------------------------------------------------- ##
    def draw_mix_dist(self):
        """
        Draw mixture distributions for each sample as in Eq. 43 of OASIS supplement.
        """
        qjs = self.rng.uniform(low = 0.0, high = 1.0, size = (self.n_targs, self.n_samps))
        qjs /= np.sum(qjs, axis = 0)
        return qjs

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
        Draw covariate arrays. 
        """
        ## equal sized groups
        x_arr = jnp.ones((self.n_samps, 1))
        ## equal sized groups
        z_arr = helpers.draw_time_arr(
            row = self.n_samps,
            col = 1,
            reg_int = True,
            rng = self.rng
        )
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
