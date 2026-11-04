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
    ## SAMPLERS                                                               ##
    ## ---------------------------------------------------------------------- ##
    def draw_covars(self):
        """
        Draw covariate arrays.

        ARGS: None
        RAISES: None
        RETURNS: two jnp.arrays --- target-specific fixed effects covariates, random effects covariates and two None
        """
        ## column names 
        x_names = []
        ## start with intercept
        x_cols = [jnp.ones((self.n_samps, 1))]
        for i in range(len(self.x_types)):
            name_i = f'fx{i}'
            if self.x_types[i] == 'cont':
                ## random noise
                x_i = helpers.draw_gaussian_arr(
                    row   = self.n_samps,
                    col   = 1,
                    mu    = self.x_mu,
                    sigma = self.x_sigma,
                    rng   = self.rng,
                )
            elif self.x_types[i] == 'time_reg':
                ## linear trend/time
                x_i = helpers.draw_time_arr(
                    row = self.n_samps,
                    col = 1,
                    reg_int = True,
                    rng   = self.rng,
                )
            elif self.x_types[i] == 'time_irreg':
                ## linear trend/time
                x_i = helpers.draw_time_arr(
                    row = self.n_samps,
                    col = 1,
                    reg_int = False,
                    rng   = self.rng,
                )
            else:
                ## batch effect
                x_i, name_i = helpers.draw_factor_arr(
                    row = self.n_samps,
                    levels = int(re.sub(r"\D", "", self.x_types[i])),
                    rng   = self.rng,
                    covar = f'fx{i}'
                )
            x_cols.append(x_i)
            x_names.append(name_i)
        ## no random intercept
        z_cols = []
        z_names = []
        for i in range(len(self.z_types)):
            name_i = f'rd{i}'
            if self.z_types[i] == 'cont':
                z_i = helpers.draw_gaussian_arr(
                    row   = self.n_samps,
                    col   = 1,
                    mu    = self.z_mu,
                    sigma = self.z_sigma,
                    rng   = self.rng
                )
            elif self.z_types[i] == 'time_reg':
                ## linear trend/time --- regular intervals
                z_i = helpers.draw_time_arr(
                    row = self.n_samps,
                    col = 1,
                    reg_int = True,
                    rng   = self.rng,
                )
            elif self.z_types[i] == 'time_irreg':
                ## linear trend/time --- irregular intervals
                z_i = helpers.draw_time_arr(
                    row = self.n_samps,
                    col = 1,
                    reg_int = False,
                    rng   = self.rng,
                )
            else:
                z_i, name_i = helpers.draw_factor_arr(
                    row = self.n_samps,
                    levels = int(re.sub(r"\D", "", self.z_types[i])),
                    rng   = self.rng,
                    covar = f'rd{i}'
                )
            z_cols.append(z_i)
            z_names.append(name_i)
        x_arr = jnp.concatenate(x_cols, axis = 1)
        z_arr = jnp.concatenate(z_cols, axis = 1)
        return x_arr, z_arr, x_names, z_names

    def draw_effs(self) -> jnp.array:
        """Draw coefficients.
        
        ARGS: None
        RAISES: None
        RETURNS: two jnp.arrays --- target-specific fixed effects coefficients, random effects coefficients
        """
        fx_effs = helpers.draw_gaussian_arr(
            row   = self.n_targs,
            col   = self.x_arr.shape[-1], ## total number of fixed effects
            mu    = self.fx_effs_mu,
            sigma = self.fx_effs_sigma,
            rng   = self.rng,
        )
        rd_effs = helpers.draw_gaussian_arr(
            row   = self.n_targs,
            col   = self.z_arr.shape[-1], ## total number of random effects
            mu    = self.rd_effs_mu,
            sigma = self.rd_effs_sigma,
            rng   = self.rng,
        )
        return fx_effs, rd_effs

    # def draw_col_sums(self):
    #     """
    #     Draw random column (sample counts) sums.

    #     ARGS:    None
    #     RAISES:  None
    #     RETURNS: two jnp.arrays; (n_samps,) and (n_samps,)
    #     """
    #     if self.col_sums_low == self.col_sums_high:
    #         col_sums = jnp.ones((self.n_samps,)) * self.col_sums_low
    #         offsets = jnp.zeros((self.n_samps,))
    #         return col_sums, offsets 
    #     ## sample column sums
    #     col_sums = jnp.array(
    #         self.rng.integers(
    #             low  = self.col_sums_low,
    #             high = self.col_sums_high,
    #             size = self.n_samps
    #         )
    #     )
    #     # log(col_sums) - mean(log(col_sums))
    #     offsets = jnp.log(col_sums) / jnp.exp(jnp.mean(jnp.log(col_sums)))
    #     return col_sums, offsets

        
    ## ---------------------------------------------------------------------- ##
    ## COMPUTATIONS                                                           ##
    ## ---------------------------------------------------------------------- ##
    def compute_eta(self) -> jnp.array:
        """
        Compute "true" linear predictor.

        ARGS:    None
        RAISES:  None
        RETURNS: jnp.array; linear predictors (n_targs, n_samps)
        """
        fx_part = self.fx_effs_arr @ self.x_arr.T
        rd_part = self.rd_effs_arr @ self.z_arr.T
        lin_pred = fx_part + rd_part
        # lin_pred += jnp.tile(self.offsets, (self.n_targs, 1))
        return lin_pred
    
    ## ---------------------------------------------------------------------- ##
    ## SETTERS                                                                ##
    ## ---------------------------------------------------------------------- ##
    def update_n_samps(self, n_samps):
        """Update number of samples by removing data. Attributes that depend upon the number of samples include: n_samps, x_arr, z_arr, eta_arr, mu_arr, counts, targ_idx, samp_idx, column sums.
        """
        ## new number of samples is smaller than before so subset data
        ## uncenter the data because it will be re-centered in set_covars()
        new_x, new_z, new_x_names, new_z_names = self.get_covars(ctr = False)
        self.set_covars(new_x[:n_samps, :], new_z[:n_samps, :], new_x_names, new_z_names)
        # new_col_sums = self.col_sums[:n_samps]
        # new_offsets  = self.offsets[:n_samps]
        # self.set_col_sums(new_col_sums, new_offsets)
        targ_arr  = np.array(self.targ_idx)
        samp_arr  = np.array(self.samp_idx)
        count_arr = np.array(self.counts)
        keep = np.where(samp_arr <= n_samps)
        new_targ_idx = targ_arr[keep].tolist()
        new_samp_idx = samp_arr[keep].tolist()
        new_counts   = count_arr[keep].tolist()
        self.set_tab_vals(new_targ_idx, new_samp_idx, new_counts)
        new_y = self.construct_y()
        rel_targs = new_y.sum(axis = 1) > 0
        rel_samps = new_y.sum(axis = 0) > 0
        self.set_rel_idx(rel_targs, rel_samps)
        ## set dimensions
        self.n_samps = n_samps

    def update_n_targs(self, n_targs):
        """Update number of targets by removing data. Attributes that depend upon the number of targets include: n_targs, fx_effs_arr, rd_effs_arr, eta_arr, mu_arr, counts, targ_idx, samp_idx.
        """
        ## new number of targets is smaller than before so subset data
        new_fx, new_rd = self.get_effs()
        self.set_effs(new_fx[:n_targs, :], new_rd[:n_targs, :])
        targ_arr  = np.array(self.targ_idx)
        samp_arr  = np.array(self.samp_idx)
        count_arr = np.array(self.counts)
        keep = np.where(targ_arr <= n_targs)
        new_targ_idx = targ_arr[keep].tolist()
        new_samp_idx = samp_arr[keep].tolist()
        new_counts   = count_arr[keep].tolist()
        self.set_tab_vals(new_targ_idx, new_samp_idx, new_counts)
        new_y = self.construct_y()
        rel_targs = new_y.sum(axis = 1) > 0
        rel_samps = new_y.sum(axis = 0) > 0
        self.set_rel_idx(rel_targs, rel_samps)
        ## dispersion parameter
        self.set_disp_arr(self.disp_arr[:n_targs, :])
        ## set dimensions
        self.n_targs = n_targs

    ## ---------------------------------------------------------------------- ##
    ## CHILD CLASS FUNCTIONS                                                  ##
    ## ---------------------------------------------------------------------- ##
    def sample(self):
        raise NotImplementedError("Subclasses must implement this method.")


    # def draw_offsets(self):
    #     raise NotImplementedError("Subclasses must implement this method.")
        