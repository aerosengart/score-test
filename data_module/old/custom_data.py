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
        ## mean and standard deviation for fixed effects
        self.fx_effs_mu      = float(config['fx_effs']['mu'])
        self.fx_effs_sigma   = float(config['fx_effs']['sigma'])
        ## mean and standard deviation for fixed effects covariates
        if 'mu' in config['x'].keys() and 'sigma' in config['x'].keys():
            self.x_mu      = float(config['x']['mu'])
            self.x_sigma   = float(config['x']['sigma'])
        self.x_types   = config['x']['types']
        ## mean and standard deviation for random effects covariates
        if 'mu' in config['z'].keys() and 'sigma' in config['z'].keys():
            self.z_mu    = float(config['z']['mu'])
            self.z_sigma = float(config['z']['sigma'])
        self.z_types   = config['z']['types']
        ## mean and standard deviation for dispersion
        if 'disp' in config.keys():
            self.disp_mu    = float(config['disp']['mu'])
            self.disp_sigma = float(config['disp']['sigma'])
        ## set var_comp
        self.rd_effs_mu = float(config['rd_effs']['mu'])
        self.var_comp = jnp.array(config['var_comp'])
        
        ## initialize parent class
        super().__init__(config)


    ## ---------------------------------------------------------------------- ##
    ## COMPUTATIONS                                                           ##
    ## ---------------------------------------------------------------------- ##
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
        lin_pred += jnp.tile(self.offsets, (self.n_targs, 1))
        return lin_pred
    
    ## ---------------------------------------------------------------------- ##
    ## SAMPLERS                                                               ##
    ## ---------------------------------------------------------------------- ##
    def update_n_samps(self, n_samps):
        """Update number of targets and draw additional variables. Attributes that depend upon the number of saples include: n_samps, x_arr, z_arr, eta_arr, mu_arr, counts, targ_idx, samp_idx, column sums.
        """
        if self.n_samps < n_samps:
            ## new number of samples is bigger than before
            ## must draw additional data
            diff = n_samps - self.n_samps
            ## update number of samples to be difference in previous and current
            self.n_samps = diff
            ## update coefficient arrays 
            x_arr, z_arr = self.draw_covars()
            ## uncenter because they will be recentered with added data
            old_x, old_z = self.get_covars(ctr = False)
            new_x = jnp.concatenate(
                [old_x, x_arr], axis = 0
            )
            new_z = jnp.concatenate(
                [old_z, z_arr], axis = 0
            )
            self.set_covars(new_x, new_z)
            ## update column sums
            col_sums = self.draw_col_sums()
            new_col_sums = jnp.concatenate(
                [self.col_sums, col_sums]
            )
            self.col_sums = new_col_sums
            ## update counts --- requires constructing dense arrays
            targ_idx, samp_idx, counts = self.draw_tab_vals()
            y_arr = helpers.sparse_rep_to_table(targ_idx, samp_idx, counts)
            ## cut off redrawn samples 
            new_y = jnp.concatenate(
                [self.construct_y(), y_arr[:, -diff:]],
                axis = 1
            )
            new_targ, new_samp, new_counts = helpers.table_to_sparse_rep(new_y)
            self.set_tab_vals(new_targ, new_samp, new_counts)
            self.set_rel_idx()
        elif self.n_samps > n_samps:
            ## new number of samples is smaller than before so subset data
            ## uncenter the data because it will be re-centered in set_covars()
            new_x, new_z = self.get_covars(ctr = False)
            self.set_covars(new_x[:n_samps, :], new_z[:n_samps, :])
            self.col_sums = self.col_sums[:n_samps]
            targ_arr  = np.array(self.targ_idx)
            samp_arr  = np.array(self.samp_idx)
            count_arr = np.array(self.counts)
            keep = np.where(samp_arr <= n_samps)
            new_targ_idx = targ_arr[keep].tolist()
            new_samp_idx = samp_arr[keep].tolist()
            new_counts   = count_arr[keep].tolist()
            self.set_tab_vals(new_targ_idx, new_samp_idx, new_counts)
            self.set_rel_idx()
        ## set dimensions
        self.n_samps = n_samps

    # def update_n_targs(self, n_targs):
    #     """Update number of targets and draw additional variables. Attributes that depend upon the number of targets include: n_targs, fx_effs_arr, rd_effs_arr, disp_arr, eta_arr, mu_arr, counts, targ_idx, samp_idx.
    #     """
    #     old_n_targs = self.n_targs
    #     if self.n_targs < n_targs:
    #         ## new number of targs is bigger than before
    #         ## must draw additional samples
    #         diff = n_targs - self.n_targs
    #         ## update number of targets
    #         self.n_targs = n_targs
    #         ## update coefficient arrays 
    #         fx_effs, rd_effs = self.draw_effs()
    #         new_fx_arr = jnp.concatenate(
    #             [self.fx_effs_arr, fx_effs[:diff, :]], axis = 0
    #         )
    #         new_rd_arr = jnp.concatenate(
    #             [self.rd_effs_arr, rd_effs[:diff, :]], axis = 0
    #         )
    #         self.set_effs(new_fx_arr, new_rd_arr)
    #         ## update dispersion parameters
    #         disps = self.draw_disp()
    #         new_disps = jnp.concatenate(
    #             [self.disp_arr, disps[:diff, :]], axis = 0
    #         )
    #         self.set_disp(new_disps)
    #         ## update counts
    #         targ_idx, samp_idx, counts = self.draw_tab_vals()
    #         targ_arr  = np.array(targ_idx)
    #         samp_arr  = np.array(samp_idx)
    #         count_arr = np.array(counts)
    #         added = np.where(targ_arr > old_n_targs)
    #         new_targ_idx = self.targ_idx + targ_arr[added].tolist()
    #         new_samp_idx = self.samp_idx + samp_arr[added].tolist()
    #         new_counts   = self.counts   + count_arr[added].tolist()
    #         self.set_tab_vals(new_targ_idx, new_samp_idx, new_counts)
    #     elif self.n_targs > n_targs:
    #         ## new number of targets is smaller than before
    #         ## can just subset existing data
    #         self.n_targs = n_targs
    #         self.set_effs(self.fx_effs_arr[:n_targs, :], self.rd_effs_arr[:n_targs, :])
    #         self.set_disp(self.disp_arr[:n_targs, :])
    #         targ_arr = np.array(self.targ_idx)
    #         samp_arr = np.array(self.samp_idx)
    #         count_arr = np.array(self.counts)
    #         keep = np.where(targ_arr <= n_targs)
    #         new_targ_idx = targ_arr[keep].tolist()
    #         new_samp_idx = samp_arr[keep].tolist()
    #         new_counts   = count_arr[keep].tolist()
    #         self.set_tab_vals(new_targ_idx, new_samp_idx, new_counts)


    ## ---------------------------------------------------------------------- ##
    ## CHILD CLASS FUNCTIONS                                                  ##
    ## ---------------------------------------------------------------------- ##
    def sample(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def set_dist_attrs(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def draw_disp(self):
        raise NotImplementedError("Subclasses must implement this method.")
