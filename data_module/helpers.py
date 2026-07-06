"""
Functions to:
    - draw simulated data
    - ingest SPLASH-specific data from files
"""

import os

import jax.numpy  as jnp
import pandas     as pd
import numpy      as np
import scipy.sparse   as sps


## ---------------------------------------------------------------------- ##
## SAMPLERS                                                               ##
## ---------------------------------------------------------------------- ##
def draw_gaussian_arr(row, col, rng, mu = 0.0, sigma = 0.0):
    """
    Draw an array of i.i.d. Gaussian random variables with the provided dimensions, mean, and standard deviation.
    
    ARGS:
        row   - int; number of rows
        col   - int; number of columns
        mu    - float; mean
        sigma - float; standard deviation
        rng   - numpy random number generator
    RAISES: None
    RETURNS: jnp.array (row, col)
    """
    return jnp.array(rng.normal(
        loc   = mu,
        scale = sigma,
        size  = (row, col)
    ))


def draw_factor_arr(row, col, levels, rng):
    """
    Draw an array of factor level indicators of the specified shape. Values of the returned array will be integers sampled from {0, 1, 2, ..., levels - 1}.
    
    ARGS:
        row    - int; number of rows
        col    - int; number of columns
        levels - int; number of levels of factor
        rng    - numpy random number generator
    RAISES: None
    RETURNS: jnp.array (row, col)
    """
    level_list = range(levels)
    x_levels = jnp.asarray(rng.choice(level_list, size = row * col))
    return jnp.reshape(x_levels, (row, col))


def draw_time_arr(row, col, reg_int, rng):
    """
    Draw an array of covariates representing sampling/collection times. Values of the returned array will be floats in [0, 1]. The times are NOT ordered by column index.

    ARGS:
        row     - int; number of rows
        col     - int; number of columns
        reg_int - bool; whether "times" should be at regular intervals
        rng     - numpy random number generator
    RAISES:  None
    RETURNS: jnp.array (row, col)
    """
    arr = np.empty((row, col))
    for i in range(col):
        if reg_int:
            vals = [x / (row - 1) for x in list(range(row))]
            vals = rng.choice(vals, size = len(vals), replace = False)
            arr[:, i] = vals
        else:
            arr[:, i] = rng.uniform(0, 1, row)
    return jnp.array(arr)


## ---------------------------------------------------------------------- ##
## COMPUTATIONS                                                           ##
## ---------------------------------------------------------------------- ##
def table_to_sparse_rep(counts):
    """
    Construct lists for CSC representation of count table.

    ARGS:    counts - jnp.array; anchor count table 
    RAISES:  None
    RETURNS: three lists of integers; target indices, sample indices, and non-zero counts
    """
    ## get indices of non-zero elements in table
    non_zero = np.nonzero(counts)
    ## R is 1-indexed
    targ_idx = [1 + int(x) for x in non_zero[0]]
    samp_idx = [1 + int(x) for x in non_zero[1]]
    counts  = counts[non_zero].tolist()
    return targ_idx, samp_idx, counts

def sparse_rep_to_table(row_idx, col_idx, counts):
    """
    Construct dense anchor count table from CSC sparse representation.

    ARGS:    
        row_idx - list of ints; row indices of non-zero counts
        col_idx - list of ints; column indices of non-zero counts
        counts  - list of ints; non-zero counts
    RAISES:  None
    RETURNS: jnp.array (n_row, n_col); dense count table
    """
    ## 1-indexed
    n_row = max(row_idx)
    n_col = max(col_idx)
    coo_arr = sps.coo_array(
        (np.array(counts), 
        (np.array(row_idx) - 1, np.array(col_idx) - 1)), ## convert to 0-indexed
        (n_row, n_col)
    )
    return coo_arr.tocsc().todense()

def satc_to_sparse_rep(
        anchor:     str, 
        satc_dir:   str
    ):
    """
    Create necessary lists for sparse representation of anchor table from SaTC file.

    ARGS:
        anchor     - str;  anchor 
        satc_dir   - str;  directory of anchor-specific SaTC files
    RAISES: None
    RETURNS: 
        three lists of ints; row index, column index, and counts for non-zero cells in the anchor contingency table
        two dictionaries; target sequence-index and sample ID-index keys
    """
    anchor_file = os.path.join(satc_dir, f'{anchor}.txt')
    ## convert sample ID and target to col/row indices --- R is 1-indexed
    samp_idx_key = {}
    samp_idx_val = 1
    targ_idx_key = {}
    targ_idx_val = 1
    ## i,j,x for sparseMatrix
    row_idx = []
    col_idx = []
    tab_val = []
    with open(anchor_file, 'r') as f:
        for line in f:
            vals = line.split(' ')
            samp = str(vals[0])
            targ = str(vals[1])
            cnt  = int(vals[2])
            ## add to index dictionaries if new sample or target
            if targ not in targ_idx_key:
                targ_idx_key[targ] = targ_idx_val
                targ_idx_val += 1
            if samp not in samp_idx_key:
                samp_idx_key[samp] = samp_idx_val
                samp_idx_val += 1
            ## target list
            row_idx.append(targ_idx_key.get(targ))
            ## sample list
            col_idx.append(samp_idx_key.get(samp))
            ## count list
            tab_val.append(cnt)
    return row_idx, col_idx, tab_val, targ_idx_key, samp_idx_key


def construct_metadata(
        fx_covars:     list, 
        rd_covars:     list, 
        metadata_file: str
    ):
    """
    Filter metadata to the specified variables.

    ARGS:
        fx_covars     - list; covariates to include as target-specific fixed effects covariates
        rd_covars     - list; covariates to include as random effects covariates
        metadata_file - str; path to clean_metadata.tsv 
    RAISES: None
    RETURNS: three pandas DataFrames 
    """ 
    ## read metadata
    md = pd.read_csv(metadata_file, sep = '\t')
    ## select relevant columns
    x_cov_arr = md[
        md.columns.intersection(['srr', 'splash_id'] + fx_covars)
    ]
    z_cov_arr       = md[
        md.columns.intersection(['srr', 'splash_id'] + rd_covars)
    ]
    ## cast types
    x_cov_arr = x_cov_arr.astype({'srr': 'str', 'splash_id': 'str'})
    z_cov_arr = z_cov_arr.astype({'srr': 'str', 'splash_id': 'str'})
    return x_cov_arr, z_cov_arr
    

def check_attrbs(data):
    """Check data attributes are compatible."""
    assert hasattr(data, 'fx_effs_mu')
    assert hasattr(data, 'fx_effs_sigma')
    assert hasattr(data, 'rd_effs_mu')
    assert hasattr(data, 'rd_effs_sigma')
    if 'cont' in data.x_types:
        assert hasattr(data, 'x_mu')
        assert hasattr(data, 'x_sigma')
    if 'cont' in data.z_types:
        assert hasattr(data, 'z_mu')
        assert hasattr(data, 'z_sigma')


