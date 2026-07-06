"""
Child class of `simul_data.Data` for Poisson outcomes in unique per-sample expression alternative.
"""


import math
import numpy           as np
import jax.numpy       as jnp

from data_module import simul_data, helpers

class Data(simul_data.Data):
    def __init__(self, config):
        print('Setting random effects covariate type to per-sample.\n')
        config['x']['types'] = []
        config['x']['center'] = False
        config['z']['types'] = [f'batch_{self.n_samps}']
        config['z']['center'] = False

        ## initialize parent class
        super().__init__(config)

        ## set covariates: (n_samps, p), (n_samps, q)
        x_arr, z_arr = self.draw_covars()
        self.set_covars(x_arr, z_arr)     ## from parent

        ## set coefficients: (n_targs, p), (n_targs, q)
        fx_effs_arr, rd_effs_arr = self.draw_effs()
        self.set_effs(fx_effs_arr, rd_effs_arr)  ## from parent

        ## set dispersion array
        disp_arr = self.draw_disp()
        self.set_disp(disp_arr)

        ## draw responses
        targ_idx, samp_idx, counts = self.get_tab_vals()
        self.set_tab_vals(targ_idx, samp_idx, counts)

    ## ---------------------------------------------------------------------- ##
    ## GETTERS / SETTERS                                                      ##
    ## ---------------------------------------------------------------------- ##


    ## ---------------------------------------------------------------------- ##
    ## COMPUTATIONS                                                           ##
    ## ---------------------------------------------------------------------- ##
    def compute_mu(self):
        """Compute and return linear predictor."""
        return jnp.exp(self.compute_eta())

    ## ---------------------------------------------------------------------- ##
    ## SAMPLERS                                                               ##
    ## ---------------------------------------------------------------------- ##
    def draw_covars(self):
        """
        Draw covariate arrays.

        ARGS: None
        RAISES: None
        RETURNS: two jnp.arrays --- target-specific fixed effects covariates, random effects covariates
        """
        ## only intercept
        x_arr = jnp.ones((self.n_samps, 1))
        ## equal sized groups
        z_arr = jnp.ones((self.n_samps, 1))
        z_arr = z_arr.at[:math.floor(self.n_samps / 2), :].set(-1.0)
        return x_arr, z_arr
    
    def draw_effs(self) -> jnp.array:
        """Draw coefficients.
        
        ARGS: None
        RAISES: None
        RETURNS: two jnp.arrays --- target-specific fixed effects coefficients, random effects coefficients
        """        
        ## small baseline for all targets
        fx_effs = jnp.ones((self.n_targs, 1)) * 0.5

        ## start all at zero
        rd_effs = jnp.zeros((self.n_targs, 1))
        ## want (1-2, :n_samps/2) to have eta = 3
        ## want (3-4, n_samps/2:) to have eta = 3
        ## want the rest to be small
        rd_effs = rd_effs.at[:4, :].set(3.0)



        rd_effs = helpers.draw_gaussian_arr(
            row   = self.n_targs,
            col   = self.q,        ## q = number of random slopes
            mu    = self.rd_effs_mu,
            sigma = self.var_comp,
            rng   = self.rng,
        )
        return fx_effs, rd_effs
        
    def draw_disp(self):
        """Sample negative binomial dispersion parameter."""
        return jnp.ones((self.n_targs, 1)) * np.inf
    
    def sample(self):
        """
        Sample responses as Poisson.

        ARGS:
            self
        RETURNS:
            y_arr - jnp.array; response array (n_targs, n_samps)
        RAISES:
            None
        """
        y_arr = jnp.array(
            self.rng.poisson(lam = self.compute_mu())
        )
        ## quick fix to avoid rows of only zeroes
        while (0 in jnp.sum(y_arr, axis = 1)) or (0 in jnp.sum(y_arr, axis = 0)):
            y_arr = jnp.array(
            self.rng.poisson(lam = self.compute_mu())
            )
        return y_arr
