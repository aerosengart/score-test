import numpy as np
import scipy

from oasis_module import optim, helpers


def oasis(
        data,
        num_splits = 5, 
        train_frac = 0.25, 
        asymptotic = False
    ):
    """
    Compute OASIS test statistic and p-value bound. 

    ARGS:
        data       - instance of simul_data.Data class
        num_splits - int; number of train/test split for computing optimized f and c vectors on (default: 5)
        train_frac - float; fraction of data to use for training (default: 0.25)
        asymptotic - bool; whether to use asymptotic p-value (default: False)

    Returns:
        float; minimum p-value computed across all splits, multiplied by the number of splits (Bonferroni correction). value is capped at 1.
        two numpy.ndarrays; the f and c vectors
    """
    ## get full anchor count table
    anch_arr = data.construct_y()
    ## dimensions
    n_rows, n_cols = anch_arr.shape
    ## initialize best p-value so far
    min_p_val = 1
    ## initialize optimal c and f vectors
    c_opt, f_opt = (np.zeros(n_cols), np.zeros(n_rows))

    ## asymptotic or finite-sample method 
    if asymptotic:
        f_c_gen_method = optim.generate_asymp_c_f    
        p_val_test_method = oasis_asymp_p_val
    else:
        f_c_gen_method = optim.generate_alt_max_c_f
        p_val_test_method = oasis_finite_p_val
    
    for i in range(num_splits):
        np.random.seed(i)
        train = helpers.split_counts_colwise(
            anch_arr       = anch_arr,
            down_samp_frac = train_frac
        )
        
        c, f = f_c_gen_method(train)
        p_val, _ = p_val_test_method(anch_arr - train, c, f)
        if p_val < min_p_val:
            min_p_val = p_val
            c_opt = c
            f_opt = f
    
    return min(1, num_splits * min_p_val), f_opt, c_opt


def oasis_perp(
        data,
        num_splits:     int   = 5, 
        train_frac:     float = 0.25,
        rand_seed:      int  = 0,
        num_rand_inits: int  = 10
):
    """
    Run OASIS-perp for the dataset. Currently only supports finite-sample valid p-value bound-based OASIS results (no asymptotic p-value implemented).

    ARGS:
        data       - instance of simul_data.Data class
        num_splits - int; number of train/test split for computing optimized f and c vectors on (default: 5)
        train_frac - float; fraction of data to use for training (default: 0.25)
    RETURNS: 
        float; minimum p-value computed across all splits, multiplied by the number of splits (Bonferroni correction) (capped at 1) 
        two numpy.ndarrays; the f and c vectors
    """
    ## get full anchor count table
    anch_arr = data.construct_y()
    ## construct list of (normalized) c vectors
    c_vec_list = helpers.construct_c_vec_list(data = data)
    train = helpers.split_counts_colwise(anch_arr, train_frac)

    ## col/row sums and dimensions
    relevant_samps = train.sum(axis = 0) > 0
    ## have to restrict c_vecs to relevant samples
    c_vec_list = [c[np.ix_(relevant_samps)] for c in c_vec_list]

    ## run OASIS with orthogonal constraints to get optimal c and f vectors
    c_vec, f_vec = optim.generate_alt_max_c_f(
        anch_arr       = train,
        rand_seed      = rand_seed,
        num_rand_inits = num_rand_inits, 
        c_vec_list     = c_vec_list
    )
    ## compute p-value
    p_val, _ = oasis_finite_p_val(
        anch_arr = anch_arr - train,
        c_opt    = c_vec,
        f_opt    = f_vec
    )
    return p_val, c_vec, f_vec


def oasis_finite_p_val(anch_arr, c_opt, f_opt):
    """
    Generate p-value bound for fixed c and f on anchor count table.

    ARGS:
        anch_arr - np.array; anchor count table
        c_opt    - np.array; optimal column embedding vector (n_samps,)
        f_opt    - np.array; optimal row embedding vector (n_targs,)
    RAISES: Error if f_opt is such that no valid p-value bound can be computed
    RETURNS:
        two floats; finite-sample p-value bound and OASIS test statistic
    """
    ## replace NaNs with 0s
    c_opt = np.nan_to_num(c_opt, 0) # type: ignore
    f_opt = np.nan_to_num(f_opt, 0) # type: ignore

    ## p-value bound is 1 if c or f vector is all the same value
    if np.all(c_opt == c_opt[0]) or np.all(f_opt == f_opt[0]):
        return 1, 0

    ## normalize f_opt to be between 0 and 1
    if not (f_opt.max() <= 1) and (f_opt.min() >= 0):
        f_opt = (f_opt - f_opt.min()) / (f_opt.max() - f_opt.min())

    ## only requirement for valid p-value
    assert((f_opt.max() <= 1) and (f_opt.min() >= 0))

    ## column sums
    nj = anch_arr.sum(axis = 0)
    
    ## compute test statistic
    # stat = f_opt @ (anch_arr - anch_arr @ np.outer(anch_arr.sum(axis = 1), nj) anch_arr.sum()) @ (c_opt * nj_inv_sqrt)
    stat = helpers.compute_test_stat(anch_arr = anch_arr, c_vec = c_opt, f_vec = f_opt)

    ## table sum
    total = anch_arr.sum()

    denom = (np.linalg.norm(c_opt)**2 - (c_opt @ np.sqrt(nj))**2 / total)
    p_val = np.nan_to_num(2 * np.exp(-2 * stat**2 / denom), 1)

    cap_p_val = min(p_val, 1)
    norm_stat = 2 * stat**2 / denom
    return cap_p_val, norm_stat


def oasis_asymp_p_val(anch_arr, c_opt, f_opt):
    """
    Generate asymptotically valid (under H0) p-value for fixed c and f on anchor count table.

    ARGS:
        anch_arr - np.array; anchor count table
        c_opt    - np.array; optimal column embedding vector (n_samps,)
        f_opt    - np.array; optimal row embedding vector (n_targs,)
    RAISES: Error if f_opt is such that no valid p-value bound can be computed
    RETURNS:
        two floats; asymptotic p-value and normalized test statistic
    """
    if (c_opt == 0).all():
        return 1
    
    ## replace NaNs with 0s
    c_opt = np.nan_to_num(c_opt, 0) # type: ignore
    f_opt = np.nan_to_num(f_opt, 0) # type: ignore

    nj = anch_arr.sum(axis = 0)
    
    ## compute test statistic
    stat = helpers.compute_test_stat(
        anch_arr = anch_arr, 
        c_vec    = c_opt, 
        f_vec    = f_opt
    )

    total = anch_arr.sum()
    mu_hat = (f_opt @ anch_arr).sum() / total
    
    var_f = (f_opt - mu_hat)**2 @ anch_arr.sum(axis = 1) / anch_arr.sum()
    total_var = var_f * (np.linalg.norm(c_opt)**2 - (c_opt @ np.sqrt(nj))**2 / total)
    
    if total_var <= 0:
        return 1 ## numerical error / issue
    
    normalized_test_stat = stat / np.sqrt(total_var)
    p_val = 2 * scipy.stats.norm.cdf(-np.abs(normalized_test_stat ))
                
    return p_val, normalized_test_stat