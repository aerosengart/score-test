"""
Child class of `data.Data` for simulated datasets.
"""
import re

import jax.numpy      as jnp
import numpy          as np

from data_module import simul_data, helpers

class Data(simul_data.Data):
    def __init__(self, config):
        """Constructor for simul_data.Data class for simulated datasets. Sets all attributes of data.Data (parent class) using specification from config. 

        ARGS:
            config: 
                a .yaml file specifying the settings for sampling data
        RETURNS:
            None
        RAISES:
            None
        """
        ## initialize parent class
        super().__init__(config)

        ## outcome distribution
        self.outcome = str(config['outcome'])
        
        ## set covariates: (n_samps, p), (n_samps, q)
        x_arr, z_arr, x_names, z_names = self.draw_covars()
        self.set_covars(x_arr, z_arr, x_names, z_names)

        ## set coefficients: (n_targs, p), (n_targs, q)
        fx_effs_arr, rd_effs_arr = self.draw_effs()
        self.set_effs(fx_effs_arr, rd_effs_arr)

    ## ---------------------------------------------------------------------- ##
    ## CHILD CLASS FUNCTIONS                                                  ##
    ## ---------------------------------------------------------------------- ##
    def sample(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def update_n_samps(self, n_samps):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def update_n_targs(self, n_targs):
        raise NotImplementedError("Subclasses must implement this method.")

    def compute_eta(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def draw_covars(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def draw_effs(self): 
        raise NotImplementedError("Subclasses must implement this method.")

    def draw_col_sums(self):
        raise NotImplementedError("Subclasses must implement this method.")