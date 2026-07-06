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

        ## set file paths for data ingestion
        self.splash_dir      = config['splash_dir']
        self.satc_dir        = config['satc_dir']
        self.metadata_file   = config['metadata_file']
        self.temp_dir        = config['temp_dir']
        
        ## clean up temporary files?
        self.clean_up = config['clean_up']

        ## covariate names
        self.fx_covars = config['fx_covars']
        self.rd_covars = config['rd_covars']

        ## anchor name and variance component (under H0!)
        self.anchor   = anchor
        self.var_comp = jnp.zeros((self.q,))

        ## have to read table first to get dimensions
        targ_idx, samp_idx, counts = self.get_tab_vals()
        self.set_tab_vals(targ_idx, samp_idx, counts)
        self.targets = [targ for targ in self.targ_dict.keys()]
        self.samples = [samp for samp in self.samp_dict.keys()]
        self.n_samps = len(self.samples)
        self.n_targs = len(self.targets)

        ## set covariate arrays
        x_arr, z_arr = self.extract_covars()
        self.set_covars(x_arr, z_arr)


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
            fx_covars     = self.fx_covars,
            rd_covars     = self.rd_covars,
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
    



        ## categorical data
        # for i in range(len(self.fx_covars)):
        #     col = self.fx_covars[i]
        #     if pd.api.types.is_string_dtype(x_pd_filt[col].dtype):
        #         x_pd_filt[col] = x_pd_filt[col].astype('category').cat.codes
        #     x_pd_filt[col] = x_pd_filt[col].astype('float')




    # def set_y_arr(self):
    #     """Setter for y_arr, sample indices, and target sequences."""
    #     ## for now, don't use sparse data because we will eventually need to compute residuals
    #     # if self.sparse:
    #     #     targ_list, samp_list, val_list, targ_key, samp_key = ingest.satc_to_sparse_mat(
    #     #         anchor   = self.anchor,
    #     #         satc_dir = self.satc_dir 
    #     #     )
    #     #     self.targ_idx = targ_list
    #     #     self.samp_idx = samp_list
    #     #     self.cnt_vals = val_list
    #     #     self.targs    = [targ for targ in targ_key.keys()]
    #     #     self.samples  = [samp for samp in samp_key.keys()]
    #     y_pd = ingest.satc_to_dense_mat(
    #             anchor   = self.anchor,
    #             satc_dir = self.satc_dir 
    #         )
    #     self.y_arr   = jnp.moveaxis(jnp.expand_dims(jnp.array(y_pd), 2), 0, 1)
    #     self.targs   = y_pd.index.astype(str).tolist()
    #     self.samples = y_pd.columns.astype(str).tolist()

