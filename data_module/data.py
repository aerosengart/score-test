"""
Parent class for data object representing clustered data for GLMM variance component score test. Supports simulated data and real data child classes. Data are assumed to be organized as in SPLASH. 
"""
import numpy     as np
import jax.numpy as jnp

class Data():
    def __init__(self, config):
        ## seed for sampling/whatever
        self.seed = int(config['seed'])
        self.rng  = np.random.default_rng(self.seed)

        ## flag to center covariates
        self.ctr_x = bool(config['x']['center'])
        self.ctr_z = bool(config['z']['center'])

    ## ---------------------------------------------------------------------- ##
    ## GETTERS / SETTERS                                                      ##
    ## ---------------------------------------------------------------------- ##
    def set_covars(self, x_arr, z_arr, x_cols = None, z_cols = None):
        """Setter for covariate arrays."""
        ## column names
        if x_cols is not None:
            self.x_col_names = x_cols
        if z_cols is not None:
            self.z_col_names = z_cols
        ## column values
        if self.ctr_x:
            self.x_ctr_vals = jnp.mean(x_arr, axis = 0)
            self.x_arr = x_arr - jnp.tile(self.x_ctr_vals, (x_arr.shape[0], 1))
        else: 
            self.x_ctr_vals = jnp.zeros((x_arr.shape[1],))
            self.x_arr = x_arr
        if self.ctr_z:
            self.z_ctr_vals = jnp.mean(z_arr, axis = 0)
            self.z_arr = z_arr - jnp.tile(self.z_ctr_vals, (z_arr.shape[0], 1))
        else:
            self.z_ctr_vals = jnp.zeros((z_arr.shape[1],))
            self.z_arr = z_arr
    
    def set_tab_vals(self, targ_idx, samp_idx, counts):
        """Setter for covariate arrays."""
        self.targ_idx = targ_idx
        self.samp_idx = samp_idx
        self.counts   = counts

    def set_rel_idx(self, rel_targs, rel_samps):
        """Compute and set relevant targets (rows that do not sum to 0) and samples. Subtract 1 for 1-indexing."""
        self.rel_targs = rel_targs
        self.rel_samps = rel_samps
        self.n_rel_targs = sum(self.rel_targs)
        self.n_rel_samps = sum(self.rel_samps)

    def get_covars(self, ctr = False):
        """Getter for covariate arrays.
        
        ARGS: ctr - whether to return the centered arrays (if centered flag was set)
        RAISES: None
        RETURNS: 
            two jax.numpy arrays; x_arr and z_arr
            two jax.numpy arrays; x array column names and z array column names
        """
        if ctr:
            return self.x_arr, self.z_arr, self.x_col_names, self.z_col_names
        unctr_x = self.x_arr + jnp.tile(self.x_ctr_vals, (self.x_arr.shape[0], 1))
        unctr_z = self.z_arr + jnp.tile(self.z_ctr_vals, (self.z_arr.shape[0], 1))
        return unctr_x, unctr_z, self.x_col_names, self.z_col_names
