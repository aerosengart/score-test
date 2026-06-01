
"""
Generalized linear model fitting and coefficient extraction. Models are fit under the null hypothesis (NO RANDOM EFFECTS). All fixed effects are assumed to be TARGET-SPECIFIC, meaning a null model is fit to each target's data. 
"""

import jax.numpy       as jnp
import rpy2.robjects   as ro

from rpy2.robjects.packages import importr


def glmgampoi_fit(model):
    """
    Wrapper for glmGamPoi fit in R. Fits separate Gamma-Poisson GLMs to each target's data. Models are constructed under H0 (no random effects).

    ARGS:
        model - instance of Model class
    RAISES:
        None
    RETURNS:
        three jnp.arrays; fixed effects estimates (k, p), dispersion parameter estimates (k, 1), estiamted means (n, k)
    """
    matrix_lib = importr('Matrix')
    glmgampoi  = importr('glmGamPoi')
    ## create sparseMatrix --- targets need to be rows
    model_df = matrix_lib.sparseMatrix(
        i = ro.IntVector(model.data.targ_idx),
        j = ro.IntVector(model.data.samp_idx),
        x = ro.IntVector(model.data.counts),
    ) 
    if model.data.fx_slopes > 0:
        ## there are slopes in model, so need to create col_data object
        x_targs_dict = {}
        formula   = f'~ 1'
        for i in range(1, model.data.x_arr.shape[1]):
            x_targs_dict[f'x{i}'] = model.data.x_arr[:, i].tolist()
            formula += f' + x{i}'
        col_df = ro.DataFrame(x_targs_dict)
        ## fit model
        fit = glmgampoi.glm_gp(
            model_df, 
            design   = ro.Formula(formula), 
            col_data = col_df,
            do_cox_reid_adjustment = False, ## don't do adjustment for bias in classical MLE
            overdispersion_shrinkage = False,
            overdispersion = model.global_disp ## shared gamma or not?
        ) 
    else:
        ## fit model
        fit = glmgampoi.glm_gp(
            model_df,
            do_cox_reid_adjustment = False, ## don't do adjustment for bias in classical MLE
            overdispersion_shrinkage = False,
            overdispersion = model.global_disp ## shared gamma or not?
        ) 
    mu_arr = jnp.array(fit.rx2['Mu'])
    fx_effs, disp = extract_estims(fit)
    return fx_effs, disp, mu_arr


def extract_estims(fit):
    """
    Extract fixed effects coefficients and negative binomial overdispersion parameter estimates.

    ARGS: fit - glmGamPoi fitted object
    RAISES: None
    RETURNS: two jnp.arrays - (n_targs, p) and (n_targs, 1)
    """
    base    = importr('base')
    ## extract fixed effects
    fx_effs = jnp.array(base.summary(fit).rx2('Beta'))
    ## overdispersion is inverse of how we specify it (1 / gamma)
    gammas  = 1 / jnp.array(base.summary(fit).rx2('overdispersions'))
    gammas = jnp.expand_dims(gammas, 1)
    return fx_effs, gammas




# def fit_model(
#         model, 
#         null_model:  bool = True,
#         fx_effs_arr: jnp.array = None,
#         rd_effs_arr: jnp.array = None,
#         gamma:       jnp.array = None,
#         tau_sqr:     jnp.array = None
#     ):
#     """
#     Fit a generalized linear (mixed) model using glmmTMB.

#     ARGS: 
#         model      - instance of Model class
#         null_model - bool; whether to fit the null model or not
#     RETURNS: 
#         jnp.array; fixed effects estimates (n, k, p)
#         jnp.array; random effects modes (n, k, q)
#         jnp.array; auxiliary variable estimate (1,) 
#         jnp.array; variance components estimates (q,)
#         jnp.array; random effects variance components estimate (k,)
#         glmmTMB fitted model
#     RAISES: 
#         None
#     """
#     ## get data frame and model formula
#     data_arr, fx_cols, rd_cols = construct_data_frame(
#         model      = model, 
#         null_model = null_model
#     )
#     formula, fx_terms = model_formula_glmmtmb(
#         model   = model, 
#         fx_cols = fx_cols, 
#         rd_cols = rd_cols
#     )
#     mapping, param_values = set_map_params(
#         model       = model,
#         fx_effs_arr = fx_effs_arr,
#         gamma       = gamma,
#     )

#     ## fit model and extract parameters
#     fit = glmmtmb_fit(
#         model    = model,
#         formula  = formula,
#         data_arr = data_arr,
#         mapping  = mapping,
#         params   = param_values
#     )
#     params = extract_coefs(
#         model      = model,
#         fit        = fit,
#         null_model = null_model
#     )
#     return params['fx_effs'], params['rd_effs'], params['disp'], \
#            params['tau_sqr'], fit
    
# def extract_coefs(model, fit, null_model):
#     """
#     Wrapper for extracting coefficients from glmmTMB fit into structure matching that in Model class implementation.

#     ARGS: 
#         model      - instance of Model class
#         fit        - glmmTMB model fit
#         null_model - bool; whether to fit the null model or not
#     RAISES:
#         None
#     RETURNS:
#         dict; dictionary containing
#             fx_effs - jnp.array; fixed effects estimates (n, k, p)
#             rd_effs - jnp.array; random effects modes (n, k, q)
#             disp    - jnp.array; auxiliary variable estimate (1,) 
#             tau_sqr - jnp.array; random effects variance components estimate (q,)
#     """
#     ## importing for rpy2
#     glmmTMB = importr('glmmTMB')
#     base    = importr('base')
#     unlist  = ro.r('unlist')
#     ## fixed effects
#     fx_effs_estims = extract_fx(model, fit)
#     ## negative binomial scale or quasi poisson dispersion
#     with (ro.default_converter + pandas2ri.converter).context():
#         disp = ro.conversion.get_conversion().rpy2py(
#             glmmTMB.sigma_glmmTMB(fit)
#         )[0]
#     ## random effects conditional modes
#     if not null_model:
#         rd_effs_estims = extract_rd(model, fit)
#         var_corr = unlist(base.summary(fit).rx('varcor'))
#         with (ro.default_converter + pandas2ri.converter).context():
#             rd_sd = ro.conversion.get_conversion().rpy2py(
#                 var_corr
#             )
#     else:
#         rd_effs_estims = jnp.zeros((model.data.n, model.data.k, model.data.q))
#         rd_sd = jnp.empty((model.data.q))
#     return {'fx_effs': fx_effs_estims, 
#             'rd_effs': rd_effs_estims, 
#             'disp': jnp.array(disp), 
#             'tau_sqr': jnp.array(rd_sd**2)}



# def glmmtmb_fit(
#         model, 
#         formula: str, 
#         data_arr: jnp.array,
#         mapping = None,
#         params  = None
#     ):
#     """
#     Wrapper for glmmTMB fit in R.

#     ARGS:
#         model    - instance of Model class
#         formula  - str; formula for glmmTMB
#         data_arr - jnp.array
#     RAISES:
#         NotImplementedError - if model outcome is not one of neg_bin or quasi_poi
#     RETURNS:
#         fit - glmmTMB model fit
#     """
#     ## importing for rpy2
#     glmmTMB   = importr('glmmTMB')
#     ## convert pandas data frame to R data frame
#     with (ro.default_converter + pandas2ri.converter).context():
#         data_df = ro.conversion.get_conversion().py2rpy(data_arr)
#     ## get family
#     if model.outcome == 'neg_bin':
#         family = glmmTMB.nbinom2 ## nbinom1
#     elif model.outcome == 'quasi_poi':
#         family = glmmTMB.nbinom1 ## stats.quasipoisson
#     else:
#         raise NotImplementedError('Distribution not implemented!')
#     ## fit glmm
#     ctrl      = ro.ListVector({'iter.max': 1e5, 'eval.max': 1e5})

#     if (mapping is not None) and (params is not None):
#         fit = glmmTMB.glmmTMB(
#             formula = ro.Formula(formula), 
#             data    = data_df,
#             family  = family,
#             map     = mapping,
#             start   = params,
#             control = glmmTMB.glmmTMBControl(optCtrl = ctrl)#, parallel = 8)
#         )
#     else:
#         fit = glmmTMB.glmmTMB( 
#             formula = ro.Formula(formula), 
#             data    = data_df,
#             family  = family,
#             control = glmmTMB.glmmTMBControl(optCtrl = ctrl)#, parallel = 8)
#         )
#     convergence = fit.rx2('fit').rx2('convergence')[0]

#     if convergence == ro.rinterface.NULL:
#         convergence = 0
    
#     if convergence != 0 and model.debug != 'off':
#         doc.debug_sample(model = model)
#         doc.debug_fit(model = model, fit = fit, method = 'default')
#         if model.debug == 'full':
#             glmmTMB.diagnose(fit)
#             args = ro.ListVector({'method': 'BGFS'})
#             if (mapping is not None) and (params is not None):
#                 fit = glmmTMB.glmmTMB(
#                     formula = ro.Formula(formula), 
#                     data    = data_df,
#                     family  = family,
#                     map     = mapping,
#                     start   = params,
#                     control = glmmTMB.glmmTMBControl(optCtrl = ctrl)
#                 )
#             else:
#                 fit = glmmTMB.glmmTMB(
#                     formula = ro.Formula(formula), 
#                     data    = data_df,
#                     family  = family,
#                     control = glmmTMB.glmmTMBControl(optCtrl = ctrl)
#                 )
#             glmmTMB.diagnose(fit)
#             doc.debug_fit(model = model, fit = fit, method = 'bgfs')
#             print('Convergence problem! Check output.')
#     return fit


# def model_formula_glmmtmb(model, fx_cols = None, rd_cols = None):
#     """
#     Construct string specifying glmm model formula for glmmTMB.

#     ARGS: 
#         model   - instance of Model class
#         fx_cols - list; fixed effects columns
#         rd_cols - list; random effects columns
#     RAISES: 
#         Exception - need to at least have one fixed or random effect
#     RETURNS: 
#         str; glmm formula
#     """
#     n = model.data.n
#     k = model.data.k

#     ## base model string
#     model_str = 'y ~ 1 + '
#     ## record terms for extraction
#     fx_terms = ['(Intercept)']
#     ## dummies for cluster membership (skip first cluster so not overdetermined)
#     dummy_vars = [f'clust_{x}' for x in range(2, k + 1)]

#     ## intercepts
#     if model.data.fx_by_k_int:
#         model_str += f'{' + '.join(dummy_vars)} + '
#         fx_terms += dummy_vars
#     if not model.data.fx_int and not model.data.fx_by_k_int:
#         model_str = f'y ~ -1 + '
#         fx_terms = []

#     for col in fx_cols:
#         if 'by_k' in col:
#             clust_col = [f'{col}:{x}' for x in dummy_vars]
#             model_str += f'{' + '.join(clust_col)} + '
#             fx_terms += clust_col
#         model_str += f'{col} + '
#         fx_terms += [col]
#     if rd_cols is not None:
#         if model.data.rd_int:
#             model_str += f'diag(1 | clust) + '
#         for col in rd_cols:
#             model_str += f'diag(0 + {col} | clust) + '

#     ## cut off extra +
#     model_str = model_str[:-3]
#     return model_str, fx_terms


# def construct_data_frame(model, null_model = True):
#     """
#     Construct data frame for glmm fitting. Extracts response, fixed effects covariates, and random effects covariates from model object and concatenates into a single Pandas array. Appends a column to specify cluster membership as well as dummy variables for the clusters if the data have cluster-specific fixed effects.

#     ARGS: 
#         model - instance of Model class
#         null_model - bool; whether to fit the null model or not
#     RETURNS: 
#         pandas array; model data frame
#         list; fixed effects names
#         list; random effects names
#     RAISES: None
#     """
#     ## get data
#     x_design  = model.data.x_arr
#     z_design  = model.data.z_arr
#     resp      = model.data.y_arr
#     n         = model.data.n
#     k         = model.data.k
#     total_obs = n * k
    
#     ## remove the columns of ones for intercepts (not needed for glmmTMB)
#     if model.data.fx_int or model.data.fx_by_k_int:
#         x_design = x_design[:, :, 1:]
#     if model.data.rd_int:
#         z_design = z_design[:, :, 1:]

#     ## reshape to n x k rows
#     resp_np = jnp.reshape(
#         a     = resp,
#         shape = (total_obs, 1),
#         order = 'F'
#     )
#     x_np = jnp.reshape(
#         a     = x_design,
#         shape = (total_obs, x_design.shape[-1]),
#         order = 'F'
#     )
#     z_np = jnp.reshape(
#         a     = z_design,
#         shape = (total_obs, z_design.shape[-1]),
#         order = 'F'
#     )

#     ## add cluster indicator
#     clusters = list(range(1, k + 1))
#     cluster_idx = np.repeat(clusters, n)
#     cluster_idx_np = jnp.reshape(
#         a     = cluster_idx, 
#         shape = (total_obs, 1), 
#         order = 'F'
#     )

#     ## components in model formula
#     fx_slopes      = list(range(model.data.fx_slope))
#     fx_by_k_slopes = list(range(model.data.fx_by_k_slope))
#     rd_slopes      = list(range(model.data.rd_slope))
#     fx_slopes      = [f'fx_{str(x)}'   for x in fx_slopes]
#     fx_by_k_slopes = [f'by_k_{str(x)}' for x in fx_by_k_slopes]
#     rd_slopes      = [f'rd_{str(x)}'   for x in rd_slopes]
#     fx_cols = fx_slopes + fx_by_k_slopes
#     rd_cols = rd_slopes
    
#     ## combine into single data frame
#     if null_model:
#         rd_cols = None
#         if model.comb:
#             ## no fixed slopes --- we test these in the combined test
#             fx_cols = []
#             data_arr = pd.DataFrame(
#                 jnp.concatenate([resp_np, cluster_idx_np], axis = 1)
#             )
#             data_arr.columns = ['y'] + ['clust']
#         else:
#             data_arr = pd.DataFrame(
#                 jnp.concatenate([resp_np, x_np, cluster_idx_np], axis = 1)
#             )
#             data_arr.columns = ['y'] + fx_cols + ['clust']
#     else:
#         data_arr = pd.DataFrame(
#             jnp.concatenate([resp_np, x_np, z_np, cluster_idx_np], axis = 1)
#         )
#         data_arr.columns = ['y'] + fx_cols + rd_cols + ['clust']
#     data_arr['clust'] = data_arr['clust'].astype('int')
#     ## create dummy variables
#     data_arr_dummy = pd.get_dummies(
#         data_arr, 
#         columns    = ['clust'],
#         drop_first = True, 
#         dtype      = 'int'
#     )
#     data_arr = pd.concat([data_arr_dummy, pd.DataFrame(cluster_idx_np)], axis = 1)
#     data_arr.columns = [str(col) for col in data_arr_dummy.columns] + ['clust']
#     return data_arr, fx_cols, rd_cols


# def set_map_params(
#         model,
#         fx_effs_arr: jnp.array,
#         gamma:       jnp.array,
#     ):
#     """
#     Set mapping and starting values for fixing parameter values in glmmTMB.

#     CURRENTLY ONLY IMPLEMENTED FOR FIXING FIXED EFFECTS AND/OR GAMMA IN NEG BIN
#     AND ONLY FOR GLOBAL FIXED EFFECTS AND GLOBAL GAMMA
    
#     beta     - fixed effects
#     b        - random effects modes
#     betadisp - log of the dispersion parameter (gamma for negative binomial, phi for quasi-Poisson)
#     theta    - log std dev for random effects (tau_sqr values)

#     ARGS:
#         fx_effs_arr - jnp.array; fixed effects values (n, k, p)
#         gamma       - jnp.array; negative binomial overdispersion (n, k, 1)
#     RAISES:
#     RETURNS:
#         mapping - dictionary of what parameters we should not estimate
#         start   - dictionary of what values to hold those parameters at
#     """
#     ## fix nothing
#     if (fx_effs_arr is None) and (gamma is None):
#         return None, None
#     ## otherwise need to do some manipulations
#     mapping = {}
#     params  = {}

#     fx_effs = []
#     if fx_effs_arr is not None:
#         ## adjust for baseline group (first cluster)
#         fx_effs_adj = fx_effs_arr
#         if model.data.fx_int > 0:
#             fx_effs += [fx_effs_adj[0, 0, 0]]
#         if model.data.fx_by_k_int > 0:
#             fx_effs_baseline = jnp.transpose(jnp.tile(
#                 fx_effs_arr[:, 0, 0], (model.data.k - 1, 1)
#             ))
#             fx_effs_adj = fx_effs_adj.at[:, 1:, 0].set(fx_effs_adj[:, 1:, 0] - fx_effs_baseline)
#             fx_effs += jnp.squeeze((jnp.reshape(
#                 a     = fx_effs_adj[0, :, 0],
#                 shape = (model.data.k),
#                 order = 'C'
#             ))).tolist()
#         if model.data.fx_slope > 0:
#             ## add fixed slopes but need to check if null model includes them by looking at last dimension of passed array
#             for j in range(fx_effs_arr.shape[-1] - 1):
#                 fx_effs += [fx_effs_adj[0, 0, j + 1]]
                
#         # if model.data.fx_by_k_slope > 0:
#         #     fx_effs_baseline = jnp.transpose(jnp.tile(
#         #         fx_effs_arr[:, 0, 1:], (model.data.k - 1,)
#         #     ))
#         #     fx_effs_adj = fx_effs_adj.at[:, 1:, 1:].set(fx_effs_adj[:, 1:, 1:] - fx_effs_baseline)
#         mapping['beta'] = ro.FactorVector([ro.NA_Character] * len(fx_effs))
#         params['beta']  = ro.FloatVector(fx_effs)

#     if gamma is not None:
#         ## ONLY WORKS FOR SHARED GAMMA PARAMETER (ACROSS ALL CLUSTERS/OBS)
#         mapping['betadisp'] = ro.FactorVector([ro.NA_Character])
#         params['betadisp']  = ro.FloatVector([jnp.log(gamma[0, 0, 0])])

#     return ro.ListVector(mapping), ro.ListVector(params)


# def extract_fx(model, fit) -> jnp.array:
#     """
#     Extract the fixed effects estimates from a glmmTMB fit.

#     ARGS: 
#         model - instance of Model class
#         fit   - an instance of model.summary (from R)
#     RETURNS: 
#         jnp.array; fixed effect MLEs (n, k, p)
#     RAISES: None
#     """
#     ## R imports
#     base      = importr('base')
#     dataframe = ro.r('data.frame')

#     ## fixed effect estimates
#     fit_summ  = base.summary(fit)
#     raw_coefs = fit_summ.rx2('coefficients').rx2('cond')
#     fx_coefs  = dataframe(raw_coefs).rx('Estimate')

#     with (ro.default_converter + pandas2ri.converter).context():
#         coef_df = ro.conversion.get_conversion().rpy2py(fx_coefs)

#     ## initialize
#     fx_effs_estims = jnp.zeros((model.data.n, model.data.k, model.data.p))

#     ## accounting for intercept in indexing
#     if model.data.fx_int or model.data.fx_by_k_int:
#         fx_start = 1
#     else: 
#         fx_start = 0

#     ## loop over all coefficient estimates
#     for coef in coef_df.index:
#         if coef == '(Intercept)':
#             ## overall intercept
#             grand_value = coef_df.loc[coef].iloc[0]
#             fx_effs_estims = fx_effs_estims.at[:, :, 0].set(
#                 grand_value + fx_effs_estims[:, :, 0]
#             )
#         elif coef.startswith('clust'):
#             ## cluster-specific intercept
#             clust_idx = int(re.sub(r'\D', '', coef)) - 1
#             clust_val = coef_df.loc[coef].iloc[0]
#             fx_effs_estims = fx_effs_estims.at[:, clust_idx, 0].set(
#                 fx_effs_estims[:, clust_idx, 0] + clust_val
#             )
#         elif ':' in coef:
#             ## cluster-specific fixed slope
#             eff_clust = coef.split(':')
#             ## add 1 to account for intercept in indexing
#             eff_idx = int(re.sub(r'\D', '', eff_clust[0])) + fx_start
#             ## clusters are 1-indexed but python is 0-indexed
#             clust_idx = int(re.sub(r'\D', '', eff_clust[1])) - 1
#             clust_val = coef_df.loc[coef].iloc[0]
#             fx_effs_estims = fx_effs_estims.at[:, clust_idx, eff_idx].set(
#                 fx_effs_estims[:, clust_idx, eff_idx] + clust_val
#             )
#         elif 'by_k' in coef:
#             ## baseline cluster value for cluster-specific slope
#             eff_idx = int(re.sub(r'\D', '', coef)) + fx_start
#             eff_val = coef_df.loc[coef].iloc[0]
#             fx_effs_estims = fx_effs_estims.at[:, :, eff_idx].set(
#                 fx_effs_estims[:, :, eff_idx] + eff_val
#             )
#         else:
#             ## overall fixed slope
#             eff_idx = int(re.sub(r'\D', '', coef)) + fx_start
#             val = coef_df.loc[coef].iloc[0]
#             fx_effs_estims = fx_effs_estims.at[:, :, eff_idx].set(
#                 fx_effs_estims[:, :, eff_idx] + val
#             )
#     return fx_effs_estims



# def extract_rd(model, fit) -> jnp.array:
#     """
#     Extract the random effects conditional modes from a glmmTMB fit.

#     ARGS: 
#         model - instance of Model class
#         fit   - an instance of model.summary (from R)
#     RETURNS: 
#         jnp.array; random effect modes (n, k, q)
#     RAISES: None
#     """
#     ## R imports
#     glmmTMB   = importr('glmmTMB')
#     dataframe = ro.r('data.frame')
#     ## random effect modes
#     rd_effs = glmmTMB.ranef_glmmTMB(fit, condVar = True)
#     rd_effs_modes = dataframe(rd_effs.rx2('cond'))

#     with (ro.default_converter + pandas2ri.converter).context():
#         coef_df = ro.conversion.get_conversion().rpy2py(rd_effs_modes)

#     ## initialize
#     rd_effs_estims = jnp.zeros((model.data.n, model.data.k, model.data.q))
#     col_names = coef_df.columns

#     for eff_idx in range(len(col_names)):
#         coef = coef_df.iloc[eff_idx]
#         #eff_idx = int(re.sub(r'\D', '', coef.name))
#         vals = jnp.tile(coef.values, (model.data.n, 1))
#         rd_effs_estims = rd_effs_estims.at[:, :, eff_idx].set(
#             rd_effs_estims[:, :, eff_idx] + vals
#         )
    
#     return rd_effs_estims
