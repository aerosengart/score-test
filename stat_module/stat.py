"""Compute test statistics."""

import math
import jax
import jax.numpy    as jnp

import jax
import os

import numpy             as np
import jax.numpy         as jnp
import pandas            as pd
import matplotlib.pyplot as plt
import rpy2.robjects     as ro

from rpy2.robjects.packages import importr
from rpy2.robjects          import pandas2ri
from rpy2.robjects          import globalenv

from model_module   import glmmtmb
from routine_module import doc
from stat_module    import info, intermed, helpers, score


def compute_score_stat(model, one_sided = False):
    """
    Compute the one- or two-sided score test statistic for variance components in a GLMM.
    
    ARGS:
        model - instance of Model class
        one_sided - boolean flag; whether to compute the one-sided score test
    RETURNS:
        two float; the one and two-sided test statistics
    RAISES:
        None
    """
    q = model.data.z_arr.shape[-1]
    ## compute score
    score_val, scale_val = score.compute_score_eff(
        model, 
        scale = model.scale
    )

    info_mat = info.compute_info_eff(model, scale_val)
    info_inv = np.linalg.pinv(info_mat)
    
    ## find infimum if one-sided test. otherwise set to 0
    inf_val = 0
    if one_sided:
        try:
            inf_val = helpers.compute_infimum(
                score     = score_val,
                info_inv  = info_inv, 
                test_p    = 0,
                q         = q
            )

        except:
            raise Exception("didn't work :(")
    
    ## compute stat
    if q == 1:
        stat = score_val**2 * info_inv 
    else:
        stat = np.matmul(
            np.matmul(np.transpose(score_val), info_inv), score_val
        )
    test_stat = np.squeeze(stat - inf_val)
    return test_stat, np.squeeze(stat)



def compute_lrt_stat(
        model,
        fx_effs_arr_n: np.array = None,
        fx_effs_arr_a: np.array = None,
        disp_arr:      np.array = None
    ):
    """
    Compute the likelihood ratio test statistic for variance components in a GLMM.
    
    NOTE: The alternative parameter space is implicitly one-sided. 

    ARGS:
        model         - instance of Model class
        fx_effs_arr_n - np.array or None; fixed values for fixed effects coefficients in null model
        fx_effs_arr_a - np.array or None; fixed values for fixed effects coefficients in alternative model
        disp_arr      - np.array or None; fixed values for dispersion parameter array in null AND alternative model
    RAISES: NotImplementedError - only can run on simulated data of small enough size
    RETURNS: float; the test statistic value
    """
    if model.data.outcome == 'splash':
        raise NotImplementedError('Cannot run LRT for real data due to memory issues.')
    ## R imports
    base  = importr('base')

    ## construct model dataframe (need same one for both model fits)
    _, fx_cols_n, rd_cols_n = glmmtmb.construct_data_frame(
        model      = model, 
        null_model = True
    )
    data_arr, fx_cols_a, rd_cols_a = glmmtmb.construct_data_frame(
        model      = model, 
        null_model = False
    )
    ## get model formulae
    formula_n = glmmtmb.construct_model_formula(
        model   = model, 
        fx_cols = fx_cols_n,
        rd_cols = rd_cols_n
    )
    formula_a = glmmtmb.construct_model_formula(
        model   = model, 
        fx_cols = fx_cols_a, 
        rd_cols = rd_cols_a
    )

    ## get fixed parameters if desired
    mapping_n, params_n = glmmtmb.set_map_params(
        model       = model,
        fx_effs_arr = fx_effs_arr_n,
        disp_arr    = disp_arr
    )
    mapping_a, params_a = glmmtmb.set_map_params(
        model       = model,
        fx_effs_arr = fx_effs_arr_a,
        disp_arr    = disp_arr
    )

    ## fit models
    fit_n = glmmtmb.fit_model(
        model    = model,
        formula  = formula_n,
        data_arr = data_arr,
        mapping  = mapping_n,
        params   = params_n
    )
    fit_a = glmmtmb.fit_model(
        model    = model,
        formula  = formula_a,
        data_arr = data_arr,
        mapping  = mapping_a,
        params   = params_a
    )

    ## extract logliks
    null_loglik = float(base.summary(fit_n).rx2('logLik')[0])
    alt_loglik  = float(base.summary(fit_a).rx2('logLik')[0])
    test_stat = -2 * (null_loglik - alt_loglik)

    ## have to truncate...
    ## i think numerical issues bc usually is on the order of -1e7
    if test_stat < 0.0:
        # print('Test statistic negative', null_loglik, alt_loglik, test_stat)
        return 0.0
    return test_stat