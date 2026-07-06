"""
Child class of `data.Data` for SPLASH-type datasets. 
"""
import numpy        as np
import jax.numpy    as jnp
import scipy.sparse as sps

from data_module import data, helpers

class Data(data.Data):
    def __init__(self, config, anchor):
        ## initialize parent class
        super().__init__(config)

        self.outcome = 'splash'

        ## set file paths for data ingestion
        self.splash_dir      = config['paths']['splash_dir']
        self.satc_dir        = config['paths']['satc_dir']
        self.metadata_file   = config['paths']['metadata_file']
        self.temp_dir        = config['paths']['temp_dir']

        ## anchor name and variance component (under H0!)
        self.anchor = anchor

        ## have to read table first to get dimensions
        targ_idx, samp_idx, counts = self.get_tab_vals()
        self.set_tab_vals(targ_idx, samp_idx, counts)
        self.targets = [targ for targ in self.targ_dict.keys()]
        self.samples = [samp for samp in self.samp_dict.keys()]
        self.n_samps = len(self.samples)
        self.n_targs = len(self.targets)

        ## covariate types
        self.x_types = config['x']['types']
        self.z_types = config['z']['types']
        ## number of effects
        self.p = len(self.x_types)
        self.q = len(self.z_types)

        ## set covariate arrays
        x_arr, z_arr = self.extract_covars()
        self.set_covars(x_arr, z_arr)

    ## ---------------------------------------------------------------------- ##
    ## COMPUTATIONS                                                           ##
    ## ---------------------------------------------------------------------- ##
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
    def get_tab_vals(self):
        """
        Read anchor table from file and convert to sparse representation. Additionally sets class attributes `targ_key` and `samp_key` so that indexing with targ_idx and samp_idx can be traced back to target sequences and SPLASH sample IDs. 
        
        ARGS:   None
        RAISES: None
        RETURNS: targ_idx, samp_idx, counts --- row index list, column index list, and count list for anchor table
        """
        targ_idx, samp_idx, counts, t_dict, s_dict = helpers.satc_to_sparse_rep(
            anchor   = self.anchor,
            satc_dir = self.satc_dir
        )
        self.targ_dict = t_dict
        self.samp_dict = s_dict
        return targ_idx, samp_idx, counts

    def extract_covars(self):
        """Read covariate arrays from file."""
        ## subset metadata to variables of interest
        x_pd, z_pd = helpers.construct_metadata(
            fx_covars     = self.x_types,
            rd_covars     = self.z_types,
            metadata_file = self.metadata_file
        )

        ## filter to relevant samples in SAME ORDER as SATC file
        x_pd_filt = x_pd.set_index('splash_id').reindex(self.samples).reset_index()
        z_pd_filt = z_pd.set_index('splash_id').reindex(self.samples).reset_index()

        ## remove id columns and convert to numeric
        x_pd_filt = x_pd_filt.drop(columns = ['splash_id', 'srr'])
        z_pd_filt = z_pd_filt.drop(columns = ['splash_id', 'srr'])

        ## add intercepts
        x_arr = jnp.array(x_pd_filt)
        z_arr = jnp.array(z_pd_filt)
        if x_arr.shape[1] == 0:
            ## if no target-specific fixed slopes
            x_arr = jnp.ones((self.n_samps, 1))
        else:
            x_arr = jnp.concatenate([jnp.ones((self.n_samps, 1)), x_arr], axis = 1)

        return x_arr, z_arr
    