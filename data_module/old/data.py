"""
Parent class for data object representing clustered data for GLMM variance component score test. Supports simulated data and real data child classes. Data are assumed to be organized as in SPLASH. 

The count table is represented as three lists for sparse matrix construction: targ_idx (row indices), samp_idx (column indices), counts (cell value). Covariates are shared sample-wise across targets, so they are stored as an (n_samp, p) or (n_samp, q) arrays.

All fixed effects are assumed to be TARGET-SPECIFIC, so that the null models can be fit with glmGamPoi() (i.e. target-wise null models). Null models with ALWAYS INCLUDE a target-specific, fixed intercept. User can decide to include additional slopes.

ATTRIBUTES:
    fx_slope - int; number of target-specific fixed slopes
    rd_slope - int; number of random slopes

    targ_idx - list; targ_idx[i] is the row index of counts[i]
    samp_idx - list; samp_idx[i] is the column index of counts[i]
    counts   - list; cell values (non-negative integer)

    n_samps - int; number of samples
    n_targs - int; number of targets

    x_arr - jnp.array; target-specific fixed effects covariates 
    z_arr       - jnp.array; random effects covariates

    fx_effs_arr - jnp.array; target-specific fixed effects coefficients 
    rd_effs_arr - jnp.array; random effects coefficients

    var_comp     - jnp.array; random effects variance component(s) (q,)
    
    p - int; number of different fixed effects
    q - int; total number of random effects
"""

import jax.numpy as jnp

class Data():
    def __init__(self, config):
        """Constructor for class Data. This class organizes all of the data-related information like the response vector/array, the covariates, etc.

        Sets:
            config - dict; configuration dictionary (passed as argument)

        ARGS:
            config: 
                a .yaml file specifying the settings for sampling data
        RETURNS:
            None
        RAISES:
            None
        """
        ## for doc.py (should remove this and edit that file to read attrbs)
        self.config  = config

        self.ctr_x = False
        self.ctr_z = False
        if 'x' in config.keys():
            if 'center' in config['x'].keys():
                self.ctr_x = config['x']['center']
            if 'center' in config['z'].keys():
                self.ctr_z = config['z']['center']
                
        ## outcome type
        self.outcome = config['outcome']

        ## linear predictor construction
        if self.outcome != 'splash':
            self.fx_slopes = len(config['x']['types'])
            self.rd_slopes = len(config['z']['types'])
        else:
            self.fx_slopes = config['fx_slopes']
            self.rd_slopes = config['rd_slopes']

        ## number of fixed and random effects
        self.p = 1 + self.fx_slopes
        self.q = self.rd_slopes
        
    ## ---------------------------------------------------------------------- ##
    ## GETTERS / SETTERS                                                      ##
    ## ---------------------------------------------------------------------- ##
    def set_covars(self, x_arr, z_arr):
        """Setter for covariate arrays."""
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

    def set_rel_idx(self):
        """Compute and set relevant targets (rows that do not sum to 0) and samples."""
        self.rel_targs = list(set(self.targ_idx))
        self.rel_samps = list(set(self.samp_idx))

    def get_covars(self, ctr = False):
        """Getter for covariate arrays.
        
        ARGS: ctr - whether to return the centered arrays (if centered flag was set)
        RAISES: None
        RETURNS: two jax.numpy arrays; x_arr and z_arr
        """
        if ctr:
            return self.x_arr, self.z_arr
        unctr_x = self.x_arr + jnp.tile(self.x_ctr_vals, (self.x_arr.shape[0], 1))
        unctr_z = self.z_arr + jnp.tile(self.z_ctr_vals, (self.z_arr.shape[0], 1))
        return unctr_x, unctr_z
    ## ---------------------------------------------------------------------- ##
    ## CHILD CLASS FUNCTIONS                                                  ##
    ## ---------------------------------------------------------------------- ##
    def get_tab_vals(self):
        raise NotImplementedError("Child classes must implement this method.")
    