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
        float; the test statistic value
    RAISES:
        None
    """
    ## compute score
    score_val, scale_val = score.compute_score_var_comp(
        model, 
        scale = model.scale
    )
    ## compute information
    info_fx, info_fx_var, info_var = info.compute_info(
        model, 
        scale_val = scale_val
    )
    ## schur complement
    schur_inv = helpers.lower_right_inv(
        upp_left  = info_fx,
        middle    = info_fx_var,
        low_right = info_var
    )
    
    ## find infimum if one-sided test. otherwise set to 0
    inf_val = 0
    if one_sided:
        try:
            inf_val = helpers.compute_infimum(
                score     = score_val,
                schur_inv = schur_inv,
                test_p    = 0,
                q         = model.data.q
            )

        except:
            raise Exception("didn't work :(")
    
    ## compute stat
    if model.data.q == 1:
        stat = score_val**2 * schur_inv
    else:
        stat = jnp.matmul(
            jnp.matmul(jnp.transpose(score_val), schur_inv), score_val
        )
    test_stat = jnp.squeeze(stat - inf_val)
    return test_stat



def compute_lrt_stat(
        model,
        fx_effs_arr_n: jnp.array = None,
        fx_effs_arr_a: jnp.array = None,
        disp_arr:      jnp.array = None
    ):
    """
    Compute the likelihood ratio test statistic for variance components in a GLMM.
    
    NOTE: The alternative parameter space is implicitly one-sided. 

    ARGS:
        model         - instance of Model class
        fx_effs_arr_n - jnp.array or None; fixed values for fixed effects coefficients in null model
        fx_effs_arr_a - jnp.array or None; fixed values for fixed effects coefficients in alternative model
        disp_arr      - jnp.array or None; fixed values for dispersion parameter array in null AND alternative model
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