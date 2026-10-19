import numpy as np

from oasis_module import helpers

def alt_max(anch_arr, c_vec, c_vec_list = None):
    """
    Run alternating maximization to optimize c and f vectors. If c_vec_list is provided, then find optimal vectors with c orthogonal to vectors in c_vec_list.

    ARGS:
        anch_arr   - np.array; anchor count table (n_targs, n_samps)
        c_vec      - np.array; sample embedding vector (n_samps,)
        c_vec_list - list; list of previously found c vectors (default: None)
    RAISES: None
    RETURNS:
        two np.arrays; c and f vectors
        float; OASIS test statistic
    """
    ## if all of the coordinates of c vector are the same, perturb the first one
    if np.all(c_vec == c_vec[0]):
        c_vec[0] = -1 * c_vec[0]
        
    ## compute centered, scaled count table (X tilde)
    anch_til_arr = helpers.compute_arr_tilde(anch_arr)
    
    ## initialize
    i        = 0
    stat_old = 0
    ## iterate until convergence or failure
    while True:
        #### find optimal f for fixed c ####
        f_vec = np.sign(anch_til_arr @ c_vec)
        ## rescale f vector to be {0,1}-valued
        f_vec1 = (f_vec + 1) / 2
        f_vec2 = (1 - f_vec) / 2
        f_vec  = f_vec1
        if np.abs(f_vec2 @ anch_til_arr @ c_vec) > np.abs(f_vec1 @ anch_til_arr @ c_vec):
            f_vec = f_vec2
        
        #### find optimal c vector for fixed f vector ####
        s_j = anch_til_arr.T @ f_vec
        c_vec = s_j

        ## correct c vector to be orthogonal to all others
        if c_vec_list is not None:
            for v in c_vec_list:
                c_vec = c_vec - v @ s_j / np.linalg.norm(v, 2)**2 * v

        ## scale so c vector has unit norm
        if np.linalg.norm(c_vec) > 0:
            c_vec /= np.linalg.norm(c_vec)
        
        #### compute objective value. if fixed, stop ####
        stat = f_vec @ anch_til_arr @ c_vec
        if stat == stat_old: 
            ## will terminate once f_opt is fixed over 2 iterations
            break
        stat_old = stat
        i += 1
        ## too many iterations
        if i > 50:
            c_vec = np.zeros_like(c_vec)
            f_vec = np.zeros_like(f_vec)
            stat  = 0
            break
    return c_vec, f_vec, np.abs(stat)

def generate_alt_max_c_f(
        anch_arr,
        rand_seed:      int  = 0,
        num_rand_inits: int  = 10,
        c_vec_list = None
    ):
    """
    Generate locally-optimal (unconstrained) c and f from random initialization. If c_vec_list is provided, then optimal c vector will be found orthogonal to the vectors in this list.
    
    ARGS:
        anch_arr       - np.array; anchor count table
        rand_seed      - int; random seed for initializations
        num_rand_inits - int; number of random initializations
        c_vec_list     - list; previously found c vectors (default: None)
    RAISES: None
    RETURNS:
        two np.arrays; optimal c and f vectors
    """
    ## random initialization and extension
    np.random.seed(rand_seed)
    ## col/row sums and dimensions
    relevant_targs = anch_arr.sum(axis = 1) > 0
    relevant_samps = anch_arr.sum(axis = 0) > 0
    n_rows, n_cols = anch_arr.shape
    if relevant_targs.sum() < 2 or relevant_samps.sum() < 2:
        return np.zeros(n_cols), np.zeros(n_rows)
    ## subset cols/rows of count table to those that are not all zeros
    anch_arr = anch_arr[np.ix_(relevant_targs, relevant_samps)]

    ## initialize
    s_base = 0
    f_max  = 0
    c_max  = 0
    ## re-initialize numRandInits times
    for _ in range(num_rand_inits):
        ## randomly initialize c vector
        c_vec = np.random.choice(
            [-1, 1], size = anch_arr.shape[1]
        ).astype('float')
        ## correct c vector so that it is orthogonal to all other vectors
        if c_vec_list is not None:
            for v in c_vec_list:
                c_vec = c_vec - c_vec @ v / np.linalg.norm(v, 2)**2 * v

        ## compute optimal test statistic and c and f vectors
        c_vec, f_vec, stat = alt_max(
            anch_arr   = anch_arr, 
            c_vec      = c_vec, 
            c_vec_list = c_vec_list
        )
        ## update if better than before
        if stat > s_base: 
            f_max  = f_vec
            c_max  = c_vec
            s_base = stat
            
    ## extend to targets and samples that didn't occur previously
    f_elong = np.random.choice([0,1], size = n_rows)
    f_elong[relevant_targs] = f_max
    f_opt = f_elong
    
    c_elong = np.zeros(n_cols)
    c_elong[np.arange(n_cols)[relevant_samps]] = c_max ### fancy indexing
    c_opt = c_elong
    
    return c_opt, f_opt


def generate_asymp_c_f(anch_arr):
    """
    Generate optimal c and f vectors for the asymptotic p-value.

    NOTE: CAN THIS BE USED WITH OASIS-PERP??

    ARGS:
        anch_arr - np.array; anchor count table
    RAISES: None
    RETURNS:
        two np.arrays; optimal c and f vectors
    """
    c = np.ones(anch_arr.shape[1])
    
    ## remove rows/columns that are all zero
    zero_idx = anch_arr.sum(axis = 1) == 0
    anch_arr = anch_arr[~zero_idx]
    zero_col = anch_arr.sum(axis = 0) == 0
    anch_arr = anch_arr[:,~zero_col]

    ## empirical probability dist p over rows of anch_arr
    p = anch_arr.sum(axis = 1) / anch_arr.sum()
    anch_arr_til = helpers.compute_arr_tilde(anch_arr)

    A = np.diag(1.0 / p) @ anch_arr_til @ anch_arr_til.T

    ## set f vector to be principal eigenvector of A
    eigvals, eigvecs = np.linalg.eig(A)
    f_vec = eigvecs[:, np.argmax(eigvals)]
    ## retain only real part of f
    f_vec = np.real(f_vec)
    
    c_vec = anch_arr_til.T @ f_vec
    c_vec /= np.linalg.norm(c_vec)

    f_old = f_vec.copy()
    ### map the entries of f_old to f_vec, with the zero_idx zeroed out
    f_vec = np.zeros(len(zero_idx))
    f_vec[~zero_idx] = f_old

    c_old = c_vec.copy()
    c_vec = np.zeros(len(zero_col))
    c_vec[~zero_col] = c_old

    return c_vec, f_vec




