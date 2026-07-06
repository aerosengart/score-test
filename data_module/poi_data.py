"""
Child class of `simul_data.Data` for Poisson outcomes.
"""

import numpy           as np
import jax.numpy       as jnp

from data_module import custom_data, helpers

class Data(custom_data.Data):
    def __init__(self, config):
        ## initialize parent class
        super().__init__(config)

        ## set dispersion parameter
        disp_arr = self.draw_disp()
        self.set_disp_arr(disp_arr)
        
    ## ---------------------------------------------------------------------- ##
    ## COMPUTATIONS                                                           ##
    ## ---------------------------------------------------------------------- ##
    def compute_mu(self):
        """Compute and return linear predictor."""
        return jnp.exp(self.compute_eta())


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
        fx_dict = {
            'fx_effs_mu':    self.fx_effs_mu,
            'fx_effs_sigma': self.fx_effs_sigma,
        }
        rd_dict = {
            'rd_effs_mu':    self.rd_effs_mu,
            'rd_effs_sigma': self.rd_effs_sigma,
        }
        x_dict = {
            'center': self.ctr_x,
            'types': self.x_types,
        }
        if hasattr(self, 'x_mu'):
            x_dict['x_mu'] = self.x_mu
        if hasattr(self, 'x_sigma'):
            x_dict['x_mu'] = self.x_sigma
        z_dict = {
            'center': self.ctr_z,
            'types': self.z_types
        }
        if hasattr(self, 'z_mu'):
            z_dict['z_mu'] = self.z_mu
        if hasattr(self, 'z_sigma'):
            z_dict['z_mu'] = self.z_sigma
        param_dict = {
            'outcome': self.outcome,
            'dims': dim_dict,
            'fx_effs': fx_dict,
            'rd_effs': rd_dict,
            'x': x_dict,
            'z': z_dict
        }
        return param_dict


    ## ---------------------------------------------------------------------- ##
    ## SAMPLERS                                                               ##
    ## ---------------------------------------------------------------------- ##
    def draw_disp(self):
        """Sample negative binomial dispersion parameter."""
        return jnp.ones((self.n_targs, 1)) * np.inf
    
    def sample(self):
        """
        Sample responses as Poisson.

        ARGS:
            self
        RETURNS:
            y_arr - jnp.array; response array (n_targs, n_samps)
        RAISES:
            None
        """
        y_arr = jnp.array(self.rng.poisson(lam = self.compute_mu()))
        return y_arr
