import jax.numpy as jnp

from rpy2.robjects.packages import importr

from stat_module import stat
from model_module import glmmtmb

def test_lrt_test(simul_data, nb_model):
    data = simul_data('tests/data/nb_data_small.yaml')
    data.resample()
    model = nb_model('tests/data/nb_model.yaml', data)

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

    ## check model formulae are correct
    correct_formula_n = 'y ~ 1 + targ_2 + targ_3 + targ_4 + targ_5 + targ_6 + targ_7 + targ_8 + targ_9 + targ_10 + fx_0 + fx_0:targ_2 + fx_0:targ_3 + fx_0:targ_4 + fx_0:targ_5 + fx_0:targ_6 + fx_0:targ_7 + fx_0:targ_8 + fx_0:targ_9 + fx_0:targ_10'
    correct_formula_a = 'y ~ 1 + targ_2 + targ_3 + targ_4 + targ_5 + targ_6 + targ_7 + targ_8 + targ_9 + targ_10 + fx_0 + fx_0:targ_2 + fx_0:targ_3 + fx_0:targ_4 + fx_0:targ_5 + fx_0:targ_6 + fx_0:targ_7 + fx_0:targ_8 + fx_0:targ_9 + fx_0:targ_10 + diag(0 + rd_0 | targ)'
    assert formula_n == correct_formula_n
    assert formula_a == correct_formula_a

    ## get fixed parameters if desired
    mapping_n, params_n = glmmtmb.set_map_params(
        model = model
    )
    mapping_a, params_a = glmmtmb.set_map_params(
        model = model
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

    ## checking against anova.glmmTMB
    stats = importr('stats')
    anova = stats.anova(fit_n, fit_a, test="LRT")
    anova_val = anova.rx2('Chisq')[-1]

    ## run function
    lrt_stat = stat.compute_lrt_stat(model)

    ## check test statistic values match
    assert test_stat == anova_val
    assert lrt_stat == anova_val
