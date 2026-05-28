"""
Child class of `data.Data` for simulated datasets.
"""
import jax.numpy      as jnp
import numpy          as np
import scipy.sparse   as sps

from data_module import data, helpers


class Data(data.Data):
    def __init__(self, config):
        """Constructor for simul_data.Data class for simulated datasets. Sets all attributes of data.Data (parent class) using specification from config. Coefficients and covariates arrays are drawn as Gaussian using draw_covars(), draw_fx_effs(), and draw_rd_effs(). Sets debugging dictionary for routine tracking as well.
        
        Additionally sets the following attributes specific to simulated data:
            fx_effs_mu    - float; fixed effects coefficients mean 
            fx_effs_sigma - float; fixed effects coefficients standard deviation
            x_mu          - float; fixed effects covariates mean
            x_sigma       - float; fixed effects covariates standard deviation
            z_mu          - float; random effects covariates mean
            z_sigma       - float; random effects covariates standard deviation
            tau_sqr       - jnp.array; non-negative floats (q,)

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
        self.n_samps     = int(config['n_samps'])
        ## number of targets
        self.n_targs     = int(config['n_targs'])

        ## mean and standard deviation for fixed effects
        self.fx_effs_mu      = float(config['fx_effs']['mu'])
        self.fx_effs_sigma   = float(config['fx_effs']['sigma'])
        ## mean and standard deviation for fixed effects covariates
        self.x_mu      = float(config['x']['mu'])
        self.x_sigma   = float(config['x']['sigma'])
        ## mean and standard deviation for random effects covariates
        self.z_mu    = float(config['z']['mu'])
        self.z_sigma = float(config['z']['sigma'])
        ## set tau_sqr
        self.tau_sqr = jnp.array(config['tau_sqr'])

        ## set total number of fixed effects (accounting for target-specific)
        self.p_star = self.n_targs + self.fx_slopes * self.n_targs

        ## set covariates: (n_samps, p), (n_samps, q)
        x_arr, z_arr = self.get_covars()
        self.set_covars(x_arr, z_arr)

        ## set coefficients: (n_targs, p), (n_targs, q)
        fx_effs_arr, rd_effs_arr = self.draw_effs()
        self.set_coefs(fx_effs_arr, rd_effs_arr)

        # compute linear predictor
        self.eta_arr = self.compute_eta()

        ## draw responses
        self.set_dist_attrs(disp_param = config['disp_param'])
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
        
    ## ---------------------------------------------------------------------- ##
    ## CHILD CLASS FUNCTIONS                                                  ##
    ## ---------------------------------------------------------------------- ##
    def get_covars(self)-> jnp.array:
        """
        Draw x_share_arr, x_targs_arr, and z_arr.

        ARGS: None
        RAISES: None
        RETURNS: two jnp.arrays --- target-specific fixed effects covariates, random effects covariates
        """
        x_arr = helpers.draw_gaussian_arr(
            row   = self.n_samps,
            col   = self.fx_slopes,
            mu    = self.x_mu,
            sigma = self.x_sigma
        )
        x_arr = jnp.concatenate([jnp.ones((self.n_samps, 1)), x_arr], axis = 1)
        z_arr = helpers.draw_gaussian_arr(
            row   = self.n_samps,
            col   = self.q,       ## q = number of random slopes
            mu    = self.z_mu,
            sigma = self.z_sigma
        )
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
            sigma = self.tau_sqr
        )
        return fx_effs, rd_effs

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
    
    def set_dist_attrs(self, disp_param):
        raise NotImplementedError("Subclasses must implement this method.")

    def set_disp_param(self, disp_param):
        raise NotImplementedError("Subclasses must implement this method.")

    # def load_from_file(self, path):
    #     raise NotImplementedError("Subclasses must implement this method.")



