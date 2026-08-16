"""
Child class of `data.Data` for simulated datasets.
"""
import re

import jax.numpy      as jnp
import numpy          as np

from data_module import data, helpers

class Data(data.Data):
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

        ## initialize debugging dictionary
        self.set_debug(None, None, None, None)
        
        ## parse configuration file attributes
        self.parse_config(config)

    ## ---------------------------------------------------------------------- ##
    ## COMPUTATIONS                                                           ##
    ## ---------------------------------------------------------------------- ##
    def construct_y(self):
        """Construct anchor table as a compressed sparse column array (CSC)."""
        y_arr = helpers.sparse_rep_to_table(
            row_idx = self.targ_idx,
            col_idx = self.samp_idx,
            counts  = self.counts
        )
        return y_arr

    ## ---------------------------------------------------------------------- ##
    ## SAMPLERS                                                               ##
    ## ---------------------------------------------------------------------- ##
    def resample(self):
        """Sample new responses and update all downstream attributes."""
        counts = self.sample()
        rel_targs = counts.sum(axis = 1) > 0
        rel_samps = counts.sum(axis = 0) > 0
        targ_idx, samp_idx, counts = helpers.table_to_sparse_rep(counts)
        self.set_tab_vals(targ_idx, samp_idx, counts)
        self.set_rel_idx(rel_targs, rel_samps)



    ## ---------------------------------------------------------------------- ##
    ## GETTERS                                                                ##
    ## ---------------------------------------------------------------------- ##
    def get_effs(self):
        """Getter for coefficients arrays.
        
        ARGS: None
        RAISES: None
        RETURNS: 
            two jax.numpy arrays; fx_effs_arr and rd_effs_arr
        """
        return self.fx_effs_arr, self.rd_effs_arr

    ## ---------------------------------------------------------------------- ##
    ## SETTERS                                                                ##
    ## ---------------------------------------------------------------------- ##
    def set_disp_arr(self, disp_arr):
        """Set dispersion parameters array."""
        self.disp_arr = disp_arr

    # def set_col_sums(self, col_sums, offsets):
    #     """Setter for column sums and offsets."""
    #     self.col_sums = col_sums
    #     self.offsets  = offsets

    def set_offsets(self, offsets):
        """Setter for GLM offsets."""
        self.offsets = offsets

    def set_effs(self, fx_effs_arr, rd_effs_arr):
        """Set coefficient arrays."""
        self.fx_effs_arr = fx_effs_arr
        self.rd_effs_arr = rd_effs_arr

    def set_debug(self, iter, rep, routine, seed):
        """Setter for updating debug dictionary."""
        self.debug = {
            'iter': iter,
            'rep': rep,
            'routine': routine,
            'seed': seed,
        }

    ## ---------------------------------------------------------------------- ##
    ## CHILD CLASS FUNCTIONS                                                  ##
    ## ---------------------------------------------------------------------- ##
    def sample(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def compute_mu(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def draw_effs(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def draw_col_sums(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def draw_covars(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def parse_config(self, config):
        raise NotImplementedError("Subclasses must implement this method.")