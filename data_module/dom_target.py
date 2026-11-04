"""
Child class of `simul_data.Data` for Multinomial outcomes in a "dominant target" scenario (i.e. for each column/sample, a single row has the majority of the counts). 

The relative dominance factor is controlled by the `corruption` value in the configuration file (`corruption: 1.0` will lead to 0 counts in the non-dominant targets while `corruption: 0.0` will lead to an even distribution of counts across all targets). 

An additional value in the configuration file (`scenario`) dictates the simulation scenario, either `H_0` or `H_1`. If `H_0` is set, then the dominant targets are chosen randomly. If `H_1`, then the dominant targets coincide with the grouping specified by the `z_arr` (the batch effect). The latter case corresponds to the alternative splicing regime in Baharav et al. 
"""

import math
import random
import numpy           as np
import jax.numpy       as jnp

from data_module import scenario_data, helpers

class Data(scenario_data.Data):
    def __init__(self, config):
        ## set corruption value
        self.set_corrupt(float(config['corruption']))
        self.set_scenario(config['scenario'])
        
        ## initialize parent class
        super().__init__(config)
    
        ## setting distribution-specific parameters
        if 'disp' in config.keys():
            self.disp_mu = float(config['disp']['mu'])
            self.disp_sigma = float(config['disp']['sigma'])
        else:
            ## default setting is all the same and equal to 2.0
            self.disp_mu = 2.0
            self.disp_sigma = 0.0

        ## set dispersion parameter
        disp_arr = self.draw_disp()
        self.set_disp_arr(disp_arr)

    ## ---------------------------------------------------------------------- ##
    ## GETTERS / SETTERS                                                      ##
    ## ---------------------------------------------------------------------- ##
    def set_corrupt(self, corrupt):
        """Set corruption parameter."""
        self.corrupt = corrupt

    def set_scenario(self, scenario):
        """Set scenario for covariates (H0 or H1)."""
        self.scenario = scenario

    ## ---------------------------------------------------------------------- ##
    ## SETTERS                                                                ##
    ## ---------------------------------------------------------------------- ##
    def update_n_samps(self, n_samps):
        """Update number of samples by removing data. Attributes that depend upon the number of samples include: n_samps, x_arr, z_arr, eta_arr, mu_arr, counts, targ_idx, samp_idx, column sums.
        """
        ## new number of samples is smaller than before so subset data
        ## uncenter the data because it will be re-centered in set_covars()
        new_x, new_z, new_x_names, new_z_names = self.get_covars(ctr = False)
        ## need to be careful because samples are sorted by batch
        self.set_covars(new_x[:n_samps, :], new_z[:n_samps, :], new_x_names, new_z_names)
        new_rd = self.rd_effs_arr[:, :n_samps]
        self.set_effs(self.fx_effs_arr, new_rd)
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
        
    def get_attrbs(self):
        """Construct data attributes dictionary."""
        dim_dict = {
            'n_targs':   self.n_targs,
            'n_samps':   self.n_samps,
        }
        x_dict = {
            'center': self.ctr_x,
            'types': self.x_types
        }
        z_dict = {
            'center': self.ctr_z,
            'types': self.z_types
        }
        param_dict = {
            'outcome': self.outcome,
            'dims': dim_dict,
            'corrupt': self.corrupt,
            'x': x_dict,
            'z': z_dict,
        }
        return param_dict
    
    ## ---------------------------------------------------------------------- ##
    ## COMPUTATIONS                                                           ##
    ## ---------------------------------------------------------------------- ##
    def compute_eta(self) -> np.array:
        """
        Compute "true" linear predictor.

        ARGS:    None
        RAISES:  None
        RETURNS: np.array; linear predictors (n_targs, n_samps)
        """
        fx_part = self.fx_effs_arr @ self.x_arr.T
        rd_part = self.rd_effs_arr
        lin_pred = fx_part + rd_part
        return lin_pred
    
    ## ---------------------------------------------------------------------- ##
    ## SAMPLERS                                                               ##
    ## ---------------------------------------------------------------------- ##
    def draw_disp(self):
        """Sample negative binomial dispersion parameter."""
        disp_arr = helpers.draw_gaussian_arr(
            row   = self.n_targs,
            col   = 1,
            mu    = self.disp_mu,
            sigma = self.disp_sigma,
            rng   = self.rng,
        )
        disp_arr[disp_arr < 0.0] = 1e-4
        #disp_arr = disp_arr.at[disp_arr < 0.0].set(1e-4) ## avoid negative disp
        return disp_arr

    def draw_covars(self):
        """
        Draw covariate arrays. We assume to have only the target-specific fixed intercept and a grouping factor for the random slopes. We assume to KNOW the grouping that dictates the signal as in Fig. S6 of OASIS supplement.
        """
        ## equal sized groups
        x_arr = np.ones((self.n_samps, 1))
        x_names = ['fx_0']
        ## equal sized groups
        z_arr, _ = helpers.draw_factor_arr(
            row = self.n_samps,
            levels = 2,
            rng   = self.rng,
            covar = 'rd0'
        )
        z_names = ['rd_0', 'rd_1']
        return x_arr, z_arr, x_names, z_names
    
    def draw_effs(self) -> np.array:
        """Draw coefficients.
        
        ARGS: None
        RAISES: None
        RETURNS: two np.arrays --- target-specific fixed effects coefficients, random effects coefficients
        """
        ## need rd_effs_arr to differ by rows AND cols so do entire matrix
        dom_targ_val = math.log(self.n_targs)
        non_dom_val = dom_targ_val * self.corrupt
        dom_targ_val -= non_dom_val
        if self.scenario == 'H1':
            g0_idxs = np.where(self.z_arr.ravel() + self.z_ctr_vals == 0.0)
            g1_idxs = np.where(self.z_arr.ravel() + self.z_ctr_vals == 1.0)
            rd_effs = np.zeros((self.n_targs, self.n_samps))
            rd_effs[0, g0_idxs] = dom_targ_val
            rd_effs[1, g1_idxs] = dom_targ_val
            #rd_effs = rd_effs.at[0, g0_idxs].set(dom_targ_val)
            #rd_effs = rd_effs.at[1, g1_idxs].set(dom_targ_val)
        else:
            rd_effs = np.zeros((self.n_targs, self.n_samps))
            ## get dominant target for each column by shuffling row idxs
            dom_targ_idx = random.choices(np.arange(self.n_targs).tolist(), k = self.n_samps)
            rd_effs[dom_targ_idx, np.arange(self.n_samps).tolist()] = dom_targ_val

        fx_effs = np.ones((self.n_targs, self.x_arr.shape[-1])) * (non_dom_val)
        return fx_effs, rd_effs
        
    def sample(self):
        """
        Sample responses as multinomial.

        ARGS:    None
        RETURNS: y_arr - np.array; response array (n_targs, n_samps)
        RAISES:  None
        """
        ## softmax for p-values
        # p_vals = self.compute_eta()
        #sample = self.rng.multinomial(n = self.col_sums, pvals = p_vals.T)
        # y_arr = np.array(sample.T)
        # return y_arr
        probs = np.divide(self.disp_arr, np.add(self.compute_mu(), self.disp_arr))
        succs = self.disp_arr
        y_arr = np.array(self.rng.negative_binomial(n = succs, p = probs))
        return y_arr
