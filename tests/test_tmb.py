import jax.numpy     as jnp
import pandas        as pd
import rpy2.robjects as ro
import numpy         as np

from model_module           import glmmtmb

from rpy2.robjects.packages import importr
from rpy2.robjects          import pandas2ri

def test_construct_data_frame(simul_data, nb_model):
    data = simul_data('tests/data/nb_data_small.yaml')
    model = nb_model('tests/data/nb_model_small.yaml', data)

    ## null model
    columns = ['y', 'fx_0', 'targ_2', 'targ_3', 'targ_4', 'targ_5', 'targ_6', 'targ_7', 'targ_8', 'targ_9', 'targ_10', 'targ']
    df, fx, rd = glmmtmb.construct_data_frame(model, null_model = True)
    assert fx == ['fx_0']
    assert rd == []
    ## y + fx_0 + (number of clusters - 1) + 'targ'
    assert df.shape == (500, 12)
    assert df.columns.tolist() == columns

    ## alt model
    columns = ['y', 'fx_0', 'rd_0', 'targ_2', 'targ_3', 'targ_4', 'targ_5', 'targ_6', 'targ_7', 'targ_8', 'targ_9', 'targ_10', 'targ']
    df, fx, rd = glmmtmb.construct_data_frame(model, null_model = False)
    assert fx == ['fx_0']
    assert rd == ['rd_0']
    ## y + fx_0 + rd_0 + (number of clusters - 1) + 'targ'
    assert df.shape == (500, 13)
    assert df.columns.tolist() == columns


def test_model_formula(simul_data, nb_model):
    data = simul_data('tests/data/nb_data_small.yaml')
    model = nb_model('tests/data/nb_model.yaml', data)

    ## null model
    df, fx, rd = glmmtmb.construct_data_frame(model, null_model = True)
    formula = glmmtmb.construct_model_formula(model, fx, rd)
    correct_formula = 'y ~ 1 + targ_2 + targ_3 + targ_4 + targ_5 + targ_6 + targ_7 + targ_8 + targ_9 + targ_10 + fx_0 + fx_0:targ_2 + fx_0:targ_3 + fx_0:targ_4 + fx_0:targ_5 + fx_0:targ_6 + fx_0:targ_7 + fx_0:targ_8 + fx_0:targ_9 + fx_0:targ_10'
    assert formula == correct_formula

    ## alt model
    df, fx, rd = glmmtmb.construct_data_frame(model, null_model = False)
    formula = glmmtmb.construct_model_formula(model, fx, rd)
    correct_formula = 'y ~ 1 + targ_2 + targ_3 + targ_4 + targ_5 + targ_6 + targ_7 + targ_8 + targ_9 + targ_10 + fx_0 + fx_0:targ_2 + fx_0:targ_3 + fx_0:targ_4 + fx_0:targ_5 + fx_0:targ_6 + fx_0:targ_7 + fx_0:targ_8 + fx_0:targ_9 + fx_0:targ_10 + diag(0 + rd_0 | targ)'
    assert formula == correct_formula

def test_map_params_simple(simul_data, nb_model):
    data = simul_data('tests/data/nb_data_small.yaml')
    model = nb_model('tests/data/nb_model_small.yaml', data)
    ## overwrite fixed effects
    new_fx_effs = jnp.ones(model.fx_effs_arr.shape)
    new_fx_effs = new_fx_effs.at[1:, :].set(new_fx_effs[1:, :] + 1)
    new_disp    = jnp.ones(model.disp_arr.shape)
    mapping, params = glmmtmb.set_map_params(model, new_fx_effs, new_disp)
    with (ro.default_converter + pandas2ri.converter).context():
        mapping = ro.conversion.get_conversion().rpy2py(mapping)
        params  = ro.conversion.get_conversion().rpy2py(params)
        map_fx_effs = mapping[0]
        map_disp    = mapping[1]
        par_fx_effs = params[0]
        par_disp    = params[1]
    ## check dimensions --- fx_slopes * n_targs, 1 for global dispersion
    assert len(map_fx_effs) == 20
    assert len(map_disp)    == 1
    assert len(par_fx_effs) == 20
    assert len(par_disp)    == 1
    ## check all values are the same and NaN in mapping
    assert len(map_fx_effs.unique()) == 1
    assert len(map_disp.unique())    == 1
    assert pd.isnull(map_fx_effs[0])
    assert pd.isnull(map_disp[0])
    ## check values are correct compared to baseline
    assert len(np.unique(par_fx_effs)) == 1
    assert par_fx_effs[0] == 1 
    assert par_disp[0]    == 0 


def test_map_params(simul_data, nb_model):
    data = simul_data('tests/data/nb_data_small.yaml')
    model = nb_model('tests/data/nb_model_small.yaml', data)
    n_targs = data.n_targs
    ## overwrite fixed effects
    mapping, params = glmmtmb.set_map_params(model, model.fx_effs_arr, model.disp_arr)
    with (ro.default_converter + pandas2ri.converter).context():
        mapping = ro.conversion.get_conversion().rpy2py(mapping)
        params  = ro.conversion.get_conversion().rpy2py(params)
        map_fx_effs = mapping[0]
        map_disp    = mapping[1]
        par_fx_effs = params[0]
        par_disp    = params[1]
    base_fx_effs = np.array([par_fx_effs[0], par_fx_effs[n_targs]]) ## fx_int + fx_slope
    ## check dimensions --- fx_slopes * n_targs, 1 for global dispersion
    assert len(map_fx_effs) == 20
    assert len(map_disp)    == 1
    assert len(par_fx_effs) == 20
    assert len(par_disp)    == 1
    ## check all values are the same and NaN in mapping
    assert len(map_fx_effs.unique()) == 1
    assert len(map_disp.unique())    == 1
    assert pd.isnull(map_fx_effs[0])
    assert pd.isnull(map_disp[0])
    ## check values are correct compared to baseline
    assert par_fx_effs[0] == model.fx_effs_arr[0, 0]
    assert par_fx_effs[n_targs] == model.fx_effs_arr[0, 1]
    assert par_fx_effs[1] == model.fx_effs_arr[1, 0] - base_fx_effs[0]
    assert par_fx_effs[1 + n_targs] == model.fx_effs_arr[1, 1] - base_fx_effs[1]



def test_extract_fx_effs(simul_data, nb_model):
    data = simul_data('tests/data/nb_data_small.yaml')
    model = nb_model('tests/data/nb_model_small.yaml', data)
    fx_effs_arr = model.fx_effs_arr

    ## get data frame and model formula
    data_arr, fx_cols, rd_cols = glmmtmb.construct_data_frame(
        model      = model, 
        null_model = True
    )
    formula = glmmtmb.construct_model_formula(
        model   = model,
        fx_cols = fx_cols,
        rd_cols = rd_cols
    )
    mapping, param_values = glmmtmb.set_map_params(
        model       = model,
        fx_effs_arr = fx_effs_arr,
        disp_arr    = None
    )
    ## fit model and extract parameters
    fit = glmmtmb.fit_model(
        model    = model,
        formula  = formula,
        data_arr = data_arr,
        mapping  = mapping,
        params   = param_values
    )
    extracted = glmmtmb.extract_fx(model, fit)
    assert jnp.allclose(extracted, fx_effs_arr)


def test_extract_rd_effs(simul_data, nb_model):
    data = simul_data('tests/data/nb_data_small.yaml')
    model = nb_model('tests/data/nb_model_small.yaml', data)
    fx_effs_arr = model.fx_effs_arr

    ## get data frame and model formula
    data_arr, fx_cols, rd_cols = glmmtmb.construct_data_frame(
        model      = model, 
        null_model = True
    )
    formula = glmmtmb.construct_model_formula(
        model   = model,
        fx_cols = fx_cols,
        rd_cols = rd_cols
    )
    mapping, param_values = glmmtmb.set_map_params(
        model       = model,
        fx_effs_arr = fx_effs_arr,
        disp_arr    = None
    )
    ## fit model and extract parameters
    fit = glmmtmb.fit_model(
        model    = model,
        formula  = formula,
        data_arr = data_arr,
        mapping  = mapping,
        params   = param_values
    )
    extracted = glmmtmb.extract_fx(model, fit)
    assert jnp.allclose(extracted, fx_effs_arr)


    ## get data frame and model formula
    data_arr, fx_cols, rd_cols = glmmtmb.construct_data_frame(
        model      = model, 
        null_model = False
    )
    formula = glmmtmb.construct_model_formula(
        model   = model,
        fx_cols = fx_cols,
        rd_cols = rd_cols
    )
    mapping, param_values = glmmtmb.set_map_params(
        model       = model,
        fx_effs_arr = fx_effs_arr,
        disp_arr    = None
    )
    ## fit model and extract parameters
    fit = glmmtmb.fit_model(
        model    = model,
        formula  = formula,
        data_arr = data_arr,
        mapping  = mapping,
        params   = param_values
    )
    extracted = glmmtmb.extract_fx(model, fit)
    assert jnp.allclose(extracted, fx_effs_arr)