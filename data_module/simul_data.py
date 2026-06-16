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
        ## lower and upper bounds for column sums if including offset
        if 'col_sums' in config.keys():
            self.incl_offset = True
            self.col_sums_low = int(config['col_sums']['low'])
            self.col_sums_high = int(config['col_sums']['high'])
            self.col_sums = self.draw_column_sums()
        else:
            ## set to 1 so that offsets equal 0
            self.col_sums = jnp.ones((self.n_samps,))
            self.incl_offset = False
        ## set var_comp
        self.rd_effs_mu = float(config['rd_effs']['mu'])
        self.var_comp = jnp.array(config['var_comp'])

        ## set total number of fixed effects (accounting for target-specific)
        self.p_star = self.n_targs + self.fx_slopes * self.n_targs

        ## set covariates: (n_samps, p), (n_samps, q)
        x_arr, z_arr = self.draw_covars()
        self.set_covars(x_arr, z_arr)

        ## set coefficients: (n_targs, p), (n_targs, q)
        fx_effs_arr, rd_effs_arr = self.draw_effs()
        self.set_effs(fx_effs_arr, rd_effs_arr)

        ## set dispersion array
        disp_arr = self.draw_disp()
        self.set_disp(disp_arr)

        ## draw responses
        targ_idx, samp_idx, counts = self.get_tab_vals()
        self.set_tab_vals(targ_idx, samp_idx, counts)
        
        ## debugging dictionary
        self.debug = {
            'iter':    'default',
            'rep':     'default',
            'routine': 'default'
        }
    
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
        self.offsets = jnp.log(self.col_sums) / jnp.exp(jnp.mean(jnp.log(self.col_sums)))
        lin_pred += jnp.tile(self.offsets, (self.n_targs, 1))
        return lin_pred

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
    def set_effs(self, fx_effs_arr, rd_effs_arr):
        """Set coefficient arrays."""
        self.fx_effs_arr = fx_effs_arr
        self.rd_effs_arr = rd_effs_arr

    def set_var_comps(self, var_comp):
        """Set random effect(s) variance component(s)."""
        ## add dimension/non-negative check?
        self.var_comp = var_comp

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

    def set_seed(self, seed):
        """Setter for random seed."""
        self.seed = int(seed)
    
    ## ---------------------------------------------------------------------- ##
    ## SAMPLERS                                                               ##
    ## ---------------------------------------------------------------------- ##
    def resample(self):
        """Sample new responses and update all downstream attributes."""
        targ_idx, samp_idx, counts = self.draw_tab_vals()
        self.set_tab_vals(targ_idx, samp_idx, counts)

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
                x_i = helpers.draw_factor_arr(
                    row = self.n_samps,
                    col = 1,
                    levels = int(re.sub(r"\D", "", self.x_types[i])),
                    rng   = self.rng,
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
                    sigma = self.z_sigma,
                    rng   = self.rng,
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
                z_i = helpers.draw_factor_arr(
                    row = self.n_samps,
                    col = 1,
                    levels = int(re.sub(r"\D", "", self.z_types[i])),
                    rng   = self.rng,
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
            sigma = self.fx_effs_sigma,
            rng   = self.rng,
        )
        rd_effs = helpers.draw_gaussian_arr(
            row   = self.n_targs,
            col   = self.q,        ## q = number of random slopes
            mu    = self.rd_effs_mu,
            sigma = self.var_comp,
            rng   = self.rng,
        )
        return fx_effs, rd_effs

    def draw_column_sums(self):
        """
        Draw random column (sample counts) sums.

        ARGS:    None
        RAISES:  None
        RETURNS: jnp.array; (n_samps,)
        """
        ## sample column sums
        col_sums = jnp.array(
            self.rng.integers(
                low  = self.config['col_sums']['low'],
                high = self.config['col_sums']['high'],
                size = self.n_samps
            )
        )
        return col_sums

    def draw_tab_vals(self):
        """Get row index, column index, and count lists for sparse representation of anchor table by sampling."""
        counts = self.sample()
        return helpers.table_to_sparse_rep(counts)

    def get_tab_vals(self):
        """Get row index, column index, and count lists for sparse representation of anchor table."""
        return self.draw_tab_vals()

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
            new_x = jnp.concatenate(
                [self.x_arr, x_arr], axis = 0
            )
            new_z = jnp.concatenate(
                [self.z_arr, z_arr], axis = 0
            )
            self.set_covars(new_x, new_z)
            ## update column sums
            col_sums = self.draw_column_sums()
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
        elif self.n_samps > n_samps:
            ## new number of samples is smaller than before
            ## can just subset existing data
            self.set_covars(self.x_arr[:n_samps, :], self.z_arr[:n_samps, :])
            self.col_sums = self.col_sums[:n_samps]
            targ_arr = np.array(self.targ_idx)
            samp_arr = np.array(self.samp_idx)
            count_arr = np.array(self.counts)
            keep = np.where(samp_arr <= n_samps)
            new_targ_idx = targ_arr[keep].tolist()
            new_samp_idx = samp_arr[keep].tolist()
            new_counts   = count_arr[keep].tolist()
            self.set_tab_vals(new_targ_idx, new_samp_idx, new_counts)
        ## set dimensions
        self.n_samps = n_samps

    def update_n_targs(self, n_targs):
        """Update number of targets and draw additional variables. Attributes that depend upon the number of targets include: n_targs, fx_effs_arr, rd_effs_arr, disp_arr, eta_arr, mu_arr, counts, targ_idx, samp_idx.
        """
        old_n_targs = self.n_targs
        if self.n_targs < n_targs:
            ## new number of targs is bigger than before
            ## must draw additional samples
            diff = n_targs - self.n_targs
            ## update number of targets
            self.n_targs = n_targs
            ## update coefficient arrays 
            fx_effs, rd_effs = self.draw_effs()
            new_fx_arr = jnp.concatenate(
                [self.fx_effs_arr, fx_effs[:diff, :]], axis = 0
            )
            new_rd_arr = jnp.concatenate(
                [self.rd_effs_arr, rd_effs[:diff, :]], axis = 0
            )
            self.set_effs(new_fx_arr, new_rd_arr)
            ## update dispersion parameters
            disps = self.draw_disp()
            new_disps = jnp.concatenate(
                [self.disp_arr, disps[:diff, :]], axis = 0
            )
            self.set_disp(new_disps)
            ## update counts
            targ_idx, samp_idx, counts = self.draw_tab_vals()
            targ_arr  = np.array(targ_idx)
            samp_arr  = np.array(samp_idx)
            count_arr = np.array(counts)
            added = np.where(targ_arr > old_n_targs)
            new_targ_idx = self.targ_idx + targ_arr[added].tolist()
            new_samp_idx = self.samp_idx + samp_arr[added].tolist()
            new_counts   = self.counts   + count_arr[added].tolist()
            self.set_tab_vals(new_targ_idx, new_samp_idx, new_counts)
        elif self.n_targs > n_targs:
            ## new number of targets is smaller than before
            ## can just subset existing data
            self.n_targs = n_targs
            self.set_effs(self.fx_effs_arr[:n_targs, :], self.rd_effs_arr[:n_targs, :])
            self.set_disp(self.disp_arr[:n_targs, :])
            targ_arr = np.array(self.targ_idx)
            samp_arr = np.array(self.samp_idx)
            count_arr = np.array(self.counts)
            keep = np.where(targ_arr <= n_targs)
            new_targ_idx = targ_arr[keep].tolist()
            new_samp_idx = samp_arr[keep].tolist()
            new_counts   = count_arr[keep].tolist()
            self.set_tab_vals(new_targ_idx, new_samp_idx, new_counts)

    ## ---------------------------------------------------------------------- ##
    ## CHILD CLASS FUNCTIONS                                                  ##
    ## ---------------------------------------------------------------------- ##
    def sample(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def set_dist_attrs(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def draw_disp(self):
        raise NotImplementedError("Subclasses must implement this method.")
