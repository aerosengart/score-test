import re

import numpy           as np
import jax.numpy       as jnp
import pandas          as pd
import rpy2.robjects   as ro

from routine_module import doc

from rpy2.robjects.packages import importr
from rpy2.robjects          import pandas2ri

def fit(
        model, 
        fx_effs_arr: None,
        disp_arr:    None,
        null_model:  bool = True,
    ):
    """
    Fit a generalized linear (mixed) model using glmmTMB.

    ARGS: 
        model      - instance of Model class
        null_model - bool; whether to fit the null model or not
    RETURNS: 
        jnp.array; fixed effects estimates (n, k, p)
        jnp.array; random effects modes (n, k, q)
        jnp.array; auxiliary variable estimate (1,) 
        jnp.array; variance components estimates (q,)
        jnp.array; random effects variance components estimate (k,)
        glmmTMB fitted model
    RAISES: 
        None
    """
    ## get data frame and model formula
    data_arr, fx_cols, rd_cols = construct_data_frame(
        model      = model, 
        null_model = null_model
    )
    formula = construct_model_formula(
        model   = model,
        fx_cols = fx_cols,
        rd_cols = rd_cols
    )
    mapping, param_values = set_map_params(
        model       = model,
        fx_effs_arr = fx_effs_arr,
        disp_arr    = disp_arr
    )

    ## fit model and extract parameters
    fit = fit_model(
        model    = model,
        formula  = formula,
        data_arr = data_arr,
        mapping  = mapping,
        params   = param_values
    )
    params = extract_coefs(
        model      = model,
        fit        = fit,
        null_model = null_model
    )
    return params['fx_effs'], params['rd_effs'], params['disp'], \
           params['var_comp'], fit


def fit_model(
        model, 
        formula: str, 
        data_arr: jnp.array,
        mapping = None,
        params  = None
    ):
    """
    Wrapper for glmmTMB fit in R.

    ARGS:
        model    - instance of Model class
        formula  - str; formula for glmmTMB
        data_arr - jnp.array
    RAISES:
        NotImplementedError - if model outcome is not one of neg_bin or quasi_poi
    RETURNS:
        fit - glmmTMB model fit
    """
    ## importing for rpy2
    glmmTMB   = importr('glmmTMB')
    ## convert pandas data frame to R data frame
    with (ro.default_converter + pandas2ri.converter).context():
        data_df = ro.conversion.get_conversion().py2rpy(data_arr)
    ## get family
    if model.outcome == 'neg_bin':
        family = glmmTMB.nbinom2 ## nbinom1
    else:
        raise NotImplementedError('Distribution not implemented!')
    ## fit glmm
    ctrl      = ro.ListVector({'iter.max': 1e5, 'eval.max': 1e5})

    if (mapping is not None) and (params is not None):
        fit = glmmTMB.glmmTMB(
            formula = ro.Formula(formula), 
            data    = data_df,
            family  = family,
            map     = mapping,
            start   = params,
            control = glmmTMB.glmmTMBControl(optCtrl = ctrl)#, parallel = 8)
        )
    else:
        fit = glmmTMB.glmmTMB( 
            formula = ro.Formula(formula), 
            data    = data_df,
            family  = family,
            control = glmmTMB.glmmTMBControl(optCtrl = ctrl)#, parallel = 8)
        )

    convergence = fit.rx2('fit').rx2('convergence')[0]

    if convergence == ro.rinterface.NULL:
        convergence = 0
    
    if convergence != 0 and model.debug != 'off':
        doc.debug_sample(model = model)
        #doc.debug_fit(model = model, fit = fit, method = 'default')
        if model.debug == 'full':
            glmmTMB.diagnose(fit)
            args = ro.ListVector({'method': 'BGFS'})
            if (mapping is not None) and (params is not None):
                fit = glmmTMB.glmmTMB(
                    formula = ro.Formula(formula), 
                    data    = data_df,
                    family  = family,
                    map     = mapping,
                    start   = params,
                    control = glmmTMB.glmmTMBControl(optCtrl = ctrl)
                )
            else:
                fit = glmmTMB.glmmTMB(
                    formula = ro.Formula(formula), 
                    data    = data_df,
                    family  = family,
                    control = glmmTMB.glmmTMBControl(optCtrl = ctrl)
                )
            glmmTMB.diagnose(fit)
            #doc.debug_fit(model = model, fit = fit, method = 'bgfs')
            print('Convergence problem! Check output.')
    return fit


def construct_data_frame(model, null_model = True):
    """
    Construct data frame for glmm fitting. Extracts response, fixed effects covariates, and random effects covariates from model object and concatenates into a single Pandas array. Appends a column to specify cluster membership as well as dummy variables for the clusters if the data have cluster-specific fixed effects.

    ARGS: 
        model - instance of Model class
        null_model - bool; whether to fit the null model or not
    RAISES: None
    RETURNS: 
        pandas.array; model data frame
        two lists; fixed and random slopes column names
    """
    if model.data.outcome == 'splash':
        raise Exception('Do not run with real data! Necessary memory not available.')
    
    ## get data
    x_design  = model.data.x_arr
    z_design  = model.data.z_arr
    resp      = model.data.construct_y()
    n_samps   = model.data.n_samps
    n_targs   = model.data.n_targs
    total_obs = n_targs * n_samps
    
    ## remove the columns of ones for intercepts (not needed for glmmTMB)
    x_design = x_design[:, 1:]

    ## reshape n_targs x n_samps rows
    resp_np = jnp.reshape(
        a     = resp,
        shape = (total_obs, 1),
        order = 'F' ## column-major (Fortran-style)
    )
    ## have to tile
    x_np = jnp.tile(x_design, (n_targs, x_design.shape[-1]))
    z_np = jnp.tile(z_design, (n_targs, z_design.shape[-1]))

    ## add cluster indicator
    targets = list(range(1, n_targs + 1))
    targ_idx = np.repeat(targets, n_samps)
    targ_idx_np = jnp.reshape(
        a     = targ_idx, 
        shape = (total_obs, 1), 
        order = 'F'
    )

    ## components in model formula
    fx_slopes = [f'fx_{str(x)}' for x in list(range(model.data.p - 1))]
    
    ## combine into single data frame
    if null_model:
        ## model under H0 -> no random effects
        rd_slopes = []
        data_arr = pd.DataFrame(
            jnp.concatenate([resp_np, x_np, targ_idx_np], axis = 1)
        )
    else:
        ## model under H1 -> random effects included
        rd_slopes = [f'rd_{str(x)}' for x in list(range(model.data.q))]
        data_arr = pd.DataFrame(
            jnp.concatenate([resp_np, x_np, z_np, targ_idx_np], axis = 1)
        )
    data_arr.columns = ['y'] + fx_slopes + rd_slopes + ['targ']
    data_arr['targ'] = data_arr['targ'].astype('int')
    ## create dummy variables for targets
    data_arr_dummy = pd.get_dummies(
        data_arr, 
        columns    = ['targ'],
        drop_first = True, 
        dtype      = 'int'
    )
    data_arr = pd.concat([data_arr_dummy, pd.DataFrame(targ_idx_np)], axis = 1)
    data_arr.columns = [str(col) for col in data_arr_dummy.columns] + ['targ']
    return data_arr, fx_slopes, rd_slopes



def construct_model_formula(model, fx_cols, rd_cols):
    """
    Construct string specifying glmm model formula for glmmTMB.

    ARGS: 
        model   - instance of Model class
        fx_cols - list; fixed effects columns
        rd_cols - list; random effects columns
    RAISES: 
        Exception - need to at least have one fixed or random effect
    RETURNS: 
        str; glmm formula
    """
    ## dimensions
    n = model.data.n_samps
    k = model.data.n_targs

    ## base model string
    model_str = 'y ~ 1 + '
    ## dummies for target membership (skip first target so not overdetermined)
    dummy_vars = [f'targ_{x}' for x in range(2, k + 1)]

    ## fixed intercepts
    model_str += f'{' + '.join(dummy_vars)} + '

    ## slopes
    for col in fx_cols:
        ## first target
        model_str += f'{col} + '
        ## all other targets
        model_str += f'{' + '.join([f'{col}:{x}' for x in dummy_vars])} + '
    if rd_cols is not None:
        for col in rd_cols:
            model_str += f'diag(0 + {col} | targ) + '

    ## cut off extra +
    model_str = model_str[:-3]
    return model_str



def set_map_params(
        model,
        fx_effs_arr: jnp.array = None,
        disp_arr:    jnp.array = None,
    ):
    """
    Set mapping and starting values for fixing parameter values in glmmTMB.

    CURRENTLY ONLY IMPLEMENTED FOR FIXING FIXED EFFECTS AND/OR GAMMA IN NEG BIN
    AND ONLY FOR GLOBAL FIXED EFFECTS AND GLOBAL GAMMA
    
    beta     - fixed effects
    b        - random effects modes
    betadisp - log of the dispersion parameter (gamma for negative binomial, phi for quasi-Poisson)
    theta    - log std dev for random effects (tau_sqr values)

    ARGS:
        fx_effs_arr - jnp.array; fixed effects coefficient values (n_targs, p)
        disp_arr    - jnp.array; dispersion parameter values (n_targs, 1)
    RAISES:
    RETURNS:
        mapping - dictionary of what parameters we should not estimate
        start   - dictionary of what values to hold those parameters at
    """
    ## fix nothing
    if (fx_effs_arr is None) and (disp_arr is None):
        return None, None
    ## otherwise need to do some manipulations
    mapping = {}
    params  = {}
    fx_effs = []

    ## do fixed effects first
    if fx_effs_arr is not None:
        ## adjust for baseline group (first target)
        fx_effs_adj = fx_effs_arr
        ## tile baseline target values for all other targets
        fx_effs_baseline = jnp.tile(
            fx_effs_arr[0, :], (model.data.n_targs - 1, 1)
        )
        ## subtract off baseline target value from other targets' values
        fx_effs_adj = fx_effs_adj.at[1:, :].set(fx_effs_adj[1:, :] - fx_effs_baseline)
        ## reshape and convert to list for use in R
        fx_effs += jnp.squeeze((jnp.reshape(
            a     = fx_effs_adj,
            shape = (model.data.n_targs * fx_effs_arr.shape[-1],),
            order = 'F' ## column-major (Fortran-style)
        ))).tolist()
        ## in glmmTMB, fixed effects coefficients are called 'beta'
        mapping['beta'] = ro.FactorVector([ro.NA_Character] * len(fx_effs))
        params['beta']  = ro.FloatVector(fx_effs)

    ## do dispersion parameters
    if disp_arr is not None:
        ## cannot set map parameters if we do not have a global dispersion parameter
        if model.estim_disp != 'global':
            raise NotImplementedError('Cannot fix a global dispersion parameter value for model with target-specific parameters.')
        ## in glmmTMB, dispersion is called 'betadisp' and is on log scale
        mapping['betadisp'] = ro.FactorVector([ro.NA_Character])
        ## just take the first value since they should all be the same!
        params['betadisp']  = ro.FloatVector([jnp.log(disp_arr[0, 0])])
    return ro.ListVector(mapping), ro.ListVector(params)


def extract_fx(model, fit) -> jnp.array:
    """
    Extract the fixed effects estimates from a glmmTMB fit.

    ARGS: 
        model - instance of Model class
        fit   - an instance of model.summary (from R)
    RETURNS: 
        jnp.array; fixed effect MLEs (n, k, p)
    RAISES: None
    """
    ## R imports
    base      = importr('base')
    dataframe = ro.r('data.frame')

    ## fixed effect estimates
    fit_summ  = base.summary(fit)
    raw_coefs = fit_summ.rx2('coefficients').rx2('cond')
    fx_coefs  = dataframe(raw_coefs).rx('Estimate')

    with (ro.default_converter + pandas2ri.converter).context():
        coef_df = ro.conversion.get_conversion().rpy2py(fx_coefs)

    ## initialize
    fx_effs_estims = jnp.zeros((model.data.n_targs, model.data.p))

    ## loop over all coefficient estimates
    for coef in coef_df.index:
        if coef == '(Intercept)':
            ## overall intercept --- add to first column of fx_effs_estims
            grand_value = coef_df.loc[coef].iloc[0]
            fx_effs_estims = fx_effs_estims.at[:, 0].set(
                grand_value + fx_effs_estims[:, 0]
            )
        elif ':' in coef:
            ## cluster-specific fixed slope
            eff_clust = coef.split(':')
            ## targets are 1-indexed but python is 0-indexed
            targ_idx = int(re.sub(r'\D', '', eff_clust[0])) - 1
            ## add 1 to account for intercept in indexing
            eff_idx = int(re.sub(r'\D', '', eff_clust[1])) + 1
            targ_val = coef_df.loc[coef].iloc[0]
            fx_effs_estims = fx_effs_estims.at[targ_idx, eff_idx].set(
                fx_effs_estims[targ_idx, eff_idx] + targ_val
            )  
        elif coef.startswith('targ'):
            ## targ-specific intercept
            targ_idx = int(re.sub(r'\D', '', coef)) - 1
            targ_val = coef_df.loc[coef].iloc[0]
            fx_effs_estims = fx_effs_estims.at[targ_idx, 0].set(
                fx_effs_estims[targ_idx, 0] + targ_val
            )
        else:
            ## baseline target value for target-specific slope
            ## add 1 to account for intercept in indexing
            eff_idx = int(re.sub(r'\D', '', coef)) + 1
            eff_val = coef_df.loc[coef].iloc[0]
            fx_effs_estims = fx_effs_estims.at[:, eff_idx].set(
                fx_effs_estims[:, eff_idx] + eff_val
            )
    return fx_effs_estims



def extract_rd(model, fit) -> jnp.array:
    """
    Extract the random effects conditional modes from a glmmTMB fit.

    ARGS: 
        model - instance of Model class
        fit   - an instance of model.summary (from R)
    RETURNS: 
        jnp.array; random effect modes (n_targs, q)
    RAISES: None
    """
    ## R imports
    glmmTMB   = importr('glmmTMB')
    dataframe = ro.r('data.frame')
    ## random effect modes
    rd_effs = glmmTMB.ranef_glmmTMB(fit, condVar = True)
    rd_effs_modes = dataframe(rd_effs.rx2('cond'))

    with (ro.default_converter + pandas2ri.converter).context():
        coef_df = ro.conversion.get_conversion().rpy2py(rd_effs_modes)

    ## initialize
    rd_effs_vals = jnp.zeros((model.data.n_targs, model.data.q))
    col_names = coef_df.columns

    for eff_idx in range(len(col_names)):
        coef = coef_df.iloc[eff_idx]
        rd_effs_vals = rd_effs_vals.at[:, eff_idx].set(
            rd_effs_vals[:, eff_idx] + coef.values
        )
    
    return rd_effs_vals


def extract_coefs(model, fit, null_model):
    """
    Wrapper for extracting coefficients from glmmTMB fit into structure matching that in Model class implementation.

    ARGS: 
        model      - instance of Model class
        fit        - glmmTMB model fit
        null_model - bool; whether to fit the null model or not
    RAISES:
        None
    RETURNS:
        dict; dictionary containing
            fx_effs - jnp.array; fixed effects estimates (n, k, p)
            rd_effs - jnp.array; random effects modes (n, k, q)
            disp    - jnp.array; auxiliary variable estimate (1,) 
            tau_sqr - jnp.array; random effects variance components estimate (q,)
    """
    ## importing for rpy2
    glmmTMB = importr('glmmTMB')
    base    = importr('base')
    unlist  = ro.r('unlist')
    ## fixed effects
    fx_effs_estims = extract_fx(model, fit)
    ## negative binomial scale or quasi poisson dispersion
    with (ro.default_converter + pandas2ri.converter).context():
        disp = ro.conversion.get_conversion().rpy2py(
            glmmTMB.sigma_glmmTMB(fit)
        )[0]
    ## random effects conditional modes
    if not null_model:
        rd_effs_vals = extract_rd(model, fit)
        var_corr = unlist(base.summary(fit).rx('varcor'))
        with (ro.default_converter + pandas2ri.converter).context():
            rd_sd = ro.conversion.get_conversion().rpy2py(
                var_corr
            )
    else:
        rd_effs_vals = jnp.zeros((model.data.n_targs, model.data.q))
        rd_sd = jnp.empty((model.data.q))
    return {'fx_effs': fx_effs_estims, 
            'rd_effs': rd_effs_vals, 
            'disp': jnp.array(disp), 
            'var_comp': jnp.array(rd_sd**2)}