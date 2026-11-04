import jax.numpy as jnp
import numpy as np

from routine_module import stat_sens

def test_fx_effs_jitter(simul_data, nb_model):
    data = simul_data('tests/data/nb_data.yaml')
    data.resample()
    model = nb_model('tests/data/nb_model_use_true.yaml', data)

    ## check model is using true values
    assert np.array_equal(model.fx_effs_arr, data.fx_effs_arr)
    assert np.array_equal(model.rd_effs_arr, data.rd_effs_arr)
    assert np.array_equal(model.disp_arr, data.disp_arr)
    assert np.array_equal(model.var_comp, data.rd_effs_sigma)
    #assert np.array_equal(model.offsets, data.offsets)
    assert np.array_equal(model.mu_arr, data.compute_mu())

    ## do jitter
    stat_sens.set_fx_effs_jitter(model = model, jitter_vals = 0.1)

    ## check only the fixed effects coefficients changed (and mu)
    assert np.array_equal(model.rd_effs_arr, data.rd_effs_arr)
    assert np.array_equal(model.disp_arr, data.disp_arr)
    assert np.array_equal(model.var_comp, data.rd_effs_sigma)
    #assert np.array_equal(model.offsets, data.offsets)
    assert not np.array_equal(model.fx_effs_arr, data.fx_effs_arr)
    assert not np.array_equal(model.compute_mu(), data.compute_mu())
    ## check that value was correct added
    t_idx = 8 ## target
    ## intercept only so don't need to consider x_arr
    correct_fx_effs = data.fx_effs_arr[t_idx, :] + 0.1
    #correct_mu      = np.exp(correct_fx_effs + data.offsets[s_idx])
    assert model.fx_effs_arr[t_idx, :] == correct_fx_effs
    #assert model.compute_mu()[t_idx, s_idx] == correct_mu



def test_disp_jitter(simul_data, nb_model):
    data = simul_data('tests/data/nb_data.yaml')
    data.resample()
    model = nb_model('tests/data/nb_model_use_true.yaml', data)

    ## check model is using true values
    assert np.array_equal(model.fx_effs_arr, data.fx_effs_arr)
    assert np.array_equal(model.rd_effs_arr, data.rd_effs_arr)
    assert np.array_equal(model.disp_arr, data.disp_arr)
    assert np.array_equal(model.var_comp, data.rd_effs_sigma)
    #assert np.array_equal(model.offsets, data.offsets)
    assert np.array_equal(model.compute_mu(), data.compute_mu())

    ## do jitter
    stat_sens.set_disp_jitter(model = model, jitter_vals = 0.1)

    ## check only the fixed effects coefficients changed (and mu)
    assert np.array_equal(model.rd_effs_arr, data.rd_effs_arr)
    assert np.array_equal(model.fx_effs_arr, data.fx_effs_arr)
    assert np.array_equal(model.compute_mu(), data.compute_mu())
    assert np.array_equal(model.var_comp, data.rd_effs_sigma)
    assert not np.array_equal(model.disp_arr, data.disp_arr)
    ## check that value was correct added
    t_idx = 8 ## target
    ## intercept only so don't need to consider x_arr
    correct_disp = data.disp_arr[t_idx, :] + 0.1
    assert model.disp_arr[t_idx, :] == 0.1 #correct_disp