"""
Child class of `data.Data` for SPLASH-type datasets. 
"""
import numpy        as np
import jax.numpy    as jnp
import scipy.sparse as sps

from data_module import data, helpers

class Data(data.Data):
    def __init__(self, config, anchor, satc_dir = None):
        ## initialize parent class
        super().__init__(config)

        self.outcome = 'splash'

        ## set file paths for data ingestion
        self.splash_dir      = config['paths']['splash_dir']
        ## needed for /scratch
        if satc_dir is None:
            self.satc_dir    = config['paths']['satc_dir']
        else:
            self.satc_dir = satc_dir
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
        self.x_dummy = config['x']['dummy']
        self.z_dummy = config['z']['dummy']
        ## set covariate arrays
        x_arr, z_arr, x_cols, z_cols = self.extract_covars()
        self.set_covars(x_arr, z_arr, x_cols, z_cols)

        # self.p = self.x_arr.shape[1] + 1 ## intercept always included!
        ## dimension of random effects
        # self.q = self.z_arr.shape[1]


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
        # csc_arr = sps.coo_array(
        #     (np.array(self.counts), 
        #     (np.array(self.targ_idx) - 1, np.array(self.samp_idx) - 1)),
        #     (self.n_targs, self.n_samps)
        # ).tocsc()
        # return csc_arr
        y_arr = helpers.sparse_rep_to_table(
            row_idx = self.targ_idx,
            col_idx = self.samp_idx,
            counts  = self.counts
        )
        return y_arr
    
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
            fx_dummy      = self.x_dummy,
            rd_dummy      = self.z_dummy,
            metadata_file = self.metadata_file
        )

        ## filter to relevant samples in SAME ORDER as SATC file
        ## becomes sample by covariate arrays
        x_pd_filt = x_pd.set_index('splash_id').reindex(self.samples).reset_index()
        z_pd_filt = z_pd.set_index('splash_id').reindex(self.samples).reset_index()

        ## filter to columns that are not all zero (can happen if using dummy)
        if x_pd_filt.shape[-1] > 2:
            x_col_sums = x_pd_filt.iloc[:, 2:].sum(axis = 0)
            zero_x_col_sums = x_col_sums.index[x_col_sums == 0].tolist()
            x_pd_filt = x_pd_filt.drop(columns = zero_x_col_sums)
        if z_pd_filt.shape[-1] > 2:
            z_col_sums = z_pd_filt.iloc[:, 2:].sum(axis = 0)
            zero_z_col_sums = z_col_sums.index[z_col_sums == 0].tolist()
            z_pd_filt = z_pd_filt.drop(columns = zero_z_col_sums)

        ## remove zero sum columns
        x_cols = x_pd_filt.columns
        z_cols = z_pd_filt.columns

        ## remove id columns 
        x_pd_filt = x_pd_filt.drop(columns = ['splash_id', 'srr'])
        z_pd_filt = z_pd_filt.drop(columns = ['splash_id', 'srr'])

        ## convert to array
        x_arr = jnp.array(x_pd_filt)
        z_arr = jnp.array(z_pd_filt)

        ## add intercepts
        if x_arr.shape[1] == 0:
            ## if no target-specific fixed slopes
            x_arr = jnp.ones((self.n_samps, 1))
        else:
            x_arr = jnp.concatenate([jnp.ones((self.n_samps, 1)), x_arr], axis = 1)
        return x_arr, z_arr, x_cols, z_cols
    