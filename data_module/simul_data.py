"""
Child class of `data.Data` for simulated datasets.
"""
import re

import jax.numpy      as jnp
import numpy          as np
import scipy.sparse   as sps

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

        ## include offset in linear predictor computation?
        if 'col_sums' in config.keys():
            self.incl_offset = True
        else:
            self.incl_offset = False
        ## number of samples
        self.n_samps = int(config['n_samps'])
        ## number of targets
        self.n_targs = int(config['n_targs'])

        ## mean and standard deviation for fixed effects
        self.fx_effs_mu      = float(config['fx_effs']['mu'])
        self.fx_effs_sigma   = float(config['fx_effs']['sigma'])
        ## mean and standard deviation for fixed effects covariates
        self.x_mu      = float(config['x']['mu'])
        self.x_sigma   = float(config['x']['sigma'])
        self.x_types   = config['x']['types']
        ## mean and standard deviation for random effects covariates
        self.z_mu    = float(config['z']['mu'])
        self.z_sigma = float(config['z']['sigma'])
        self.z_types   = config['z']['types']
        ## mean and standard deviation for dispersion
        if 'disp' in config.keys():
            self.disp_mu    = float(config['disp']['mu'])
            self.disp_sigma = float(config['disp']['sigma'])
        ## lower and upper bounds for column sums
        if 'col_sums' in config.keys():
            self.col_sums_low = int(config['col_sums']['low'])
            self.col_sums_high = int(config['col_sums']['high'])
        ## set var_comp
        self.var_comp = jnp.array(config['var_comp'])

        ## set total number of fixed effects (accounting for target-specific)
        self.p_star = self.n_targs + self.fx_slopes * self.n_targs

        ## set covariates: (n_samps, p), (n_samps, q)
        x_arr, z_arr = self.draw_covars()
        self.set_covars(x_arr, z_arr)

        ## set coefficients: (n_targs, p), (n_targs, q)
        fx_effs_arr, rd_effs_arr = self.draw_effs()
        self.set_coefs(fx_effs_arr, rd_effs_arr)

        ## draw responses
        self.set_dist_attrs()
        targ_idx, samp_idx, counts = self.get_tab_vals()
        self.set_tab_vals(targ_idx, samp_idx, counts)
        
        ## debugging dictionary
        self.debug = {
            'iter':    'default',
            'rep':     'default',
            'routine': 'default'
        }
        
    ## ---------------------------------------------------------------------- ##
    ## GETTERS/SETTERS                                                        ##
    ## ---------------------------------------------------------------------- ##
    def resample(self):
        """Sample new responses and update all downstream attributes."""
        targ_idx, samp_idx, counts = self.get_tab_vals()
        self.set_tab_vals(targ_idx, samp_idx, counts)
        
    def set_coefs(self, fx_effs_arr, rd_effs_arr):
        """Set coefficient arrays."""
        self.fx_effs_arr = fx_effs_arr
        self.rd_effs_arr = rd_effs_arr

    def set_var_comps(self, var_comp):
        """Set random effect(s) variance component(s)."""
        ## add dimension/non-negative check?
        self.var_comp = var_comp
        
    ## ---------------------------------------------------------------------- ##
    ## CHILD CLASS FUNCTIONS                                                  ##
    ## ---------------------------------------------------------------------- ##
    def draw_covars(self):
        """
        Draw x_share_arr, x_targs_arr, and z_arr.

        ARGS: None
        RAISES: None
        RETURNS: two jnp.arrays --- target-specific fixed effects covariates, random effects covariates
        """
        ## start with intercept
        x_cols = [jnp.ones((self.n_samps, 1))]
        for i in range(len(self.x_types)):
            if self.x_types[i] == 'cont':
                x_i = helpers.draw_gaussian_arr(
                    row   = self.n_samps,
                    col   = 1,
                    mu    = self.x_mu,
                    sigma = self.x_sigma
                )
            else:
                x_i = helpers.draw_factor_arr(
                    row = self.n_samps,
                    col = 1,
                    levels = int(re.sub(r"\D", "", self.x_types[i]))
                )
            x_cols.append(x_i)
        ## no random intercept
        z_cols = []
        for i in range(len(self.z_types)):
            if self.z_types[i] == 'cont':
                z_i = helpers.draw_gaussian_arr(
                    row   = self.n_samps,
                    col   = 1,
                    mu    = self.z_mu,
                    sigma = self.z_sigma
                )
            else:
                z_i = helpers.draw_factor_arr(
                    row = self.n_samps,
                    col = 1,
                    levels = int(re.sub(r"\D", "", self.z_types[i]))
                )
            z_cols.append(z_i)
        x_arr = jnp.concatenate(x_cols, axis = 1)
        z_arr = jnp.concatenate(z_cols, axis = 1)
        return x_arr, z_arr

    def draw_effs(self) -> jnp.array:
        """Draw coefficients.
        
        ARGS: None
        RAISES: None
        RETURNS: two jnp.arrays --- target-specific fixed effects coefficients, random effects coefficients
        """
        fx_effs = helpers.draw_gaussian_arr(
            row   = self.n_targs,
            col   = self.p,
            mu    = self.fx_effs_mu,
            sigma = self.fx_effs_sigma
        )
        rd_effs = helpers.draw_gaussian_arr(
            row   = self.n_targs,
            col   = self.q,        ## q = number of random slopes
            mu    = 0.0,
            sigma = self.var_comp
        )
        return fx_effs, rd_effs

    def draw_column_sums(self):
        """
        Draw random column (sample counts) sums.

        ARGS:    None
        RAISES:  None
        RETURNS: jnp.array; (n_samps,)
        """
        rng   = np.random.default_rng(None)
        ## sample column sums
        col_sums = jnp.array(
            rng.integers(
                low  = self.config['col_sums']['low'],
                high = self.config['col_sums']['high'],
                size = self.n_samps
            )
        )
        return col_sums


    def get_tab_vals(self):
        """Get row index, column index, and count lists for sparse representation of anchor table."""
        counts = self.sample()
        ## get indices of non-zero elements in table
        non_zero = np.nonzero(counts)
        ## R is 1-indexed
        targ_idx = [1 + x for x in non_zero[0]]
        samp_idx = [1 + x for x in non_zero[1]]
        counts  = counts[non_zero].tolist()
        return targ_idx, samp_idx, jnp.array(counts)


    def set_debug(self, iter, rep, routine):
        """Setter for updating debug dictionary."""
        self.debug = {
            'iter': iter,
            'rep': rep,
            'routine': routine
        }

    def compute_eta(self) -> jnp.array:
        """
        Compute "true" linear predictor.

        ARGS:    None
        RAISES:  None
        RETURNS: jnp.array; linear predictor (n_targs, n_samps)
        """
        fx_part = self.fx_effs_arr @ self.x_arr.T
        rd_part = self.rd_effs_arr @ self.z_arr.T
        lin_pred = fx_part + rd_part
        if self.incl_offset:
            col_sums = self.draw_column_sums()
        else:
            ## set to 1 so that offsets equal 0
            col_sums = jnp.ones((self.n_samps,))
        self.offsets = jnp.log(col_sums) / jnp.exp(jnp.mean(jnp.log(col_sums)))
        lin_pred += jnp.tile(self.offsets, (self.n_targs, 1))
        return lin_pred

    def construct_y(self):
        """
        Construct anchor table as a compressed sparse column array (CSC).

        ARGS:
        RAISES:
        RETURNS:
        """
        coo_arr = sps.coo_array(
            (np.array(self.counts), 
            (np.array(self.targ_idx) - 1, np.array(self.samp_idx) - 1)),
            (self.n_targs, self.n_samps)
        )
        return coo_arr.tocsc()
    
    ## ---------------------------------------------------------------------- ##
    ## CHILD CLASS FUNCTIONS                                                  ##
    ## ---------------------------------------------------------------------- ##
    def sample(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def set_dist_attrs(self):
        raise NotImplementedError("Subclasses must implement this method.")

    # def load_from_file(self, path):
    #     raise NotImplementedError("Subclasses must implement this method.")



