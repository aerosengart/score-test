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

    ## ---------------------------------------------------------------------- ##
    ## SETTERS                                                                ##
    ## ---------------------------------------------------------------------- ##
    def parse_config(self, config):
        """Set attributes provided in config file."""
        ## number of samples
        self.n_samps = int(config['n_samps'])
        ## number of targets
        self.n_targs = int(config['n_targs'])

    ## ---------------------------------------------------------------------- ##
    ## CHILD CLASS FUNCTIONS                                                  ##
    ## ---------------------------------------------------------------------- ##
    def sample(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def compute_mu(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def draw_covars(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def draw_effs(self): 
        raise NotImplementedError("Subclasses must implement this method.")

    def draw_col_sums(self):
        raise NotImplementedError("Subclasses must implement this method.")