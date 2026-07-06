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

        ## number of samples
        self.n_samps = int(config['n_samps'])
        ## number of targets
        self.n_targs = int(config['n_targs'])

        ## seed for sampling
        self.seed = int(config['seed'])
        self.rng  = np.random.default_rng(self.seed)

        if 'col_sums' in config.keys():
            self.col_sums_low = int(config['col_sums']['low'])
            self.col_sums_high = int(config['col_sums']['high'])
        else:
            self.col_sums_low = 1
            self.col_sums_high = 1
        col_sums, offsets = self.draw_col_sums()
        self.set_col_sums(col_sums, offsets)

        ## set dispersion array
        disp_arr = self.draw_disp()
        self.set_disp(disp_arr)

        ## set covariates: (n_samps, p), (n_samps, q)
        x_arr, z_arr = self.draw_covars()
        self.set_covars(x_arr, z_arr)

        ## set coefficients: (n_targs, p), (n_targs, q)
        fx_effs_arr, rd_effs_arr = self.draw_effs()
        self.set_effs(fx_effs_arr, rd_effs_arr)
        
        ## debugging dictionary
        self.debug = {
            'iter':    'default',
            'rep':     'default',
            'routine': 'default'
        }

    
    ## ---------------------------------------------------------------------- ##
    ## COMPUTATIONS                                                           ##
    ## ---------------------------------------------------------------------- ##

    
    def construct_y(self):
        """Construct anchor table as a compressed sparse column array (CSC)."""
        return helpers.sparse_rep_to_table(
            row_idx = self.targ_idx,
            col_idx = self.samp_idx,
            counts  = self.counts
        )

    ## ---------------------------------------------------------------------- ##
    ## GETTERS/SETTERS                                                        ##
    ## ---------------------------------------------------------------------- ##
    def set_col_sums(self, col_sums, offsets):
        """Set column sums and offsets."""
        self.col_sums = col_sums
        self.offsets  = offsets

    def set_effs(self, fx_effs_arr, rd_effs_arr):
        """Set coefficient arrays."""
        self.fx_effs_arr = fx_effs_arr
        self.rd_effs_arr = rd_effs_arr

    def set_var_comps(self, var_comp):
        """Set random effect(s) variance component(s)."""
        ## add dimension/non-negative check?
        self.rd_effs_sigma = var_comp

    def set_disp(self, disp_arr):
        """Setter for negative binomial dispersion parameter."""
        self.disp_arr = disp_arr

    def set_debug(self, iter, rep, routine, seed):
        """Setter for updating debug dictionary."""
        self.debug = {
            'iter': iter,
            'rep': rep,
            'routine': routine,
            'seed': seed,
        }

    ## ---------------------------------------------------------------------- ##
    ## SAMPLERS                                                               ##
    ## ---------------------------------------------------------------------- ##
    def resample(self):
        """Sample new responses and update all downstream attributes."""
        targ_idx, samp_idx, counts = self.draw_tab_vals()
        self.set_tab_vals(targ_idx, samp_idx, counts)
        self.set_rel_idx()

    def draw_tab_vals(self):
        """Get row index, column index, and count lists for sparse representation of anchor table by sampling."""
        counts = self.sample()
        return helpers.table_to_sparse_rep(counts)

    def get_tab_vals(self):
        """Get row index, column index, and count lists for sparse representation of anchor table."""
        return self.draw_tab_vals()
    
    ## ---------------------------------------------------------------------- ##
    ## CHILD CLASS FUNCTIONS                                                  ##
    ## ---------------------------------------------------------------------- ##
    def compute_eta(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def draw_col_sums(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def draw_effs(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def draw_covars(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def sample(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def set_dist_attrs(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def draw_disp(self):
        raise NotImplementedError("Subclasses must implement this method.")
