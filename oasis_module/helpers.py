import numpy as np

### split contingency table into train and test data
def split_counts(
        anch_arr,
        down_samp_frac: float = 0.5
    ): 
    """
    Split anchor count table into train/test subsets.
    Code is a slight modification of https://stackoverflow.com/questions/11818215/subsample-a-matrix-python.

    ARGS:
        anch_arr       - numpy.ndarray; anchor count table (n_targs, n_samps)
        down_samp_frac - float; proportion of data to downsample to
    RAISES:  None
    RETURNS:
        numpy.ndarray; the test set (train set can be obtained by subtracting the test set from anch_arr)
    """
    keys, counts = zip(*[
        ((i, j), anch_arr[i, j])
            for i in range(anch_arr.shape[0])
            for j in range(anch_arr.shape[1])
            if anch_arr[i, j] > 0
    ])

    ## make the cumulative counts array
    counts     = np.array(counts, dtype = np.int64)
    sum_counts = np.cumsum(counts)

    ## decide how many counts to include in the sample
    frac_select = down_samp_frac
    count_select = int(sum_counts[-1] * frac_select)

    ## choose unique counts
    ind_select = sorted(np.random.choice(
        range(sum_counts[-1]), count_select, replace = False
    ))

    ## initialize vector to hold the new counts
    out_counts = np.zeros(counts.shape, dtype = np.int64)

    ## Perform basically the merge step of merge-sort
    ## finding where the counts land in the cumulative array
    i = 0
    j = 0
    while i < len(sum_counts) and j < len(ind_select):
        if ind_select[j] < sum_counts[i]:
            j += 1
            out_counts[i] += 1
        else:
            i += 1

    ## rebuild the matrix using the `keys` list from before
    out_anch_arr = np.zeros(anch_arr.shape, dtype = np.int64)
    for i in range(len(out_counts)):
        out_anch_arr[keys[i]] = out_counts[i]
    return out_anch_arr


def split_counts_colwise(
        anch_arr,
        down_samp_frac: float = 0.5
    ): 
    """
    Wrapper for split_counts() to downsample matrix column-wise rather than overall.

    ARGS:
        anch_arr       - numpy.ndarray; anchor count table (n_targs, n_samps)
        down_samp_frac - float; proportion of data to downsample to
    RAISES:  None
    RETURNS:
        numpy.ndarray; the train set (test set can be obtained by subtracting the train set from anch_arr)
    """
    out_anch_arr = np.zeros_like(anch_arr)
    for j in range(anch_arr.shape[1]):
        if anch_arr[:,j].sum() > 0:
            out_anch_arr[:, j] = split_counts(
                anch_arr = np.reshape(anch_arr[:, j], (anch_arr.shape[0], -1)),down_samp_frac = down_samp_frac
            ).flatten()
        
    return out_anch_arr

def compute_test_stat(anch_arr, c_vec, f_vec):
    anch_arr_tilde = compute_arr_tilde(arr = anch_arr)
    return f_vec @ anch_arr_tilde @ c_vec


def compute_arr_tilde(arr):
    """
    Compute centered and scaled count table. 

    ARGS:    arr - numpy.ndarray; anchor count table (n_targs, n_samps)
    RAISES:  None
    RETURNS: numpy.ndarray; centered and scaled count table based upon H0
    """
    ## expected value
    exp_arr = np.outer(arr.sum(axis = 1), arr.sum(axis = 0)) / arr.sum()
    ## column sums
    nj = arr.sum(axis = 0)
    ## 1 / sqrt(column sums)
    nj_inv_sqrt = 1.0 / np.maximum(1, np.sqrt(nj)) ## avoid divide by 0 errors
    nj_inv_sqrt[nj == 0] = 0
    return (arr - exp_arr) * nj_inv_sqrt






# ### shift and scale an input vector to be in the range [minval, maxval]
# def normalizevec(x,minval=0,maxval=1):
#     if x.max()==x.min():
#         return np.zeros_like(x)
#     x = np.array(x)
#     x01= (x-x.min())/(x.max()-x.min())
#     return x01*(maxval-minval)+minval