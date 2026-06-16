import jax.numpy as jnp

from routine_module import stat_sens

def test_fx_effs_jitter(simul_data, nb_model):
    data = simul_data('tests/data/nb_data.yaml')
    model = nb_model('tests/data/nb_model_use_true.yaml', data)

    ## check model is using true values
    assert jnp.array_equal(model.fx_effs_arr, data.fx_effs_arr)
    assert jnp.array_equal(model.rd_effs_arr, data.rd_effs_arr)
    assert jnp.array_equal(model.disp_arr, data.disp_arr)
    assert jnp.array_equal(model.var_comp, data.var_comp)
    assert jnp.array_equal(model.offsets, data.offsets)
    assert jnp.array_equal(model.mu_arr, data.compute_mu())

    ## do jitter
    stat_sens.set_fx_effs_jitter(model = model, jitter_vals = 0.1)

    ## check only the fixed effects coefficients changed (and mu)
    assert jnp.array_equal(model.rd_effs_arr, data.rd_effs_arr)
    assert jnp.array_equal(model.disp_arr, data.disp_arr)
    assert jnp.array_equal(model.var_comp, data.var_comp)
    assert jnp.array_equal(model.offsets, data.offsets)
    assert not jnp.array_equal(model.fx_effs_arr, data.fx_effs_arr)
    assert not jnp.array_equal(model.mu_arr, data.compute_mu())
    ## check that value was correct added
    t_idx = 8 ## target
    s_idx = 2 ## sample
    ## intercept only so don't need to consider x_arr
    correct_fx_effs = data.fx_effs_arr[t_idx, s_idx] + 0.1
    correct_mu      = jnp.exp(correct_fx_effs + data.offsets[s_idx])
    assert model.fx_effs_arr[t_idx, s_idx] == correct_fx_effs
    assert model.mu_arr[t_idx, s_idx] == correct_mu



def test_disp_jitter(simul_data, nb_model):
    data = simul_data('tests/data/nb_data.yaml')
    model = nb_model('tests/data/nb_model_use_true.yaml', data)

    ## check model is using true values
    assert jnp.array_equal(model.fx_effs_arr, data.fx_effs_arr)
    assert jnp.array_equal(model.rd_effs_arr, data.rd_effs_arr)
    assert jnp.array_equal(model.disp_arr, data.disp_arr)
    assert jnp.array_equal(model.var_comp, data.var_comp)
    assert jnp.array_equal(model.offsets, data.offsets)
    assert jnp.array_equal(model.mu_arr, data.compute_mu())

    ## do jitter
    stat_sens.set_disp_jitter(model = model, jitter_vals = 0.1)

    ## check only the fixed effects coefficients changed (and mu)
    assert jnp.array_equal(model.rd_effs_arr, data.rd_effs_arr)
    assert jnp.array_equal(model.fx_effs_arr, data.fx_effs_arr)
    assert jnp.array_equal(model.mu_arr, data.compute_mu())
    assert jnp.array_equal(model.var_comp, data.var_comp)
    assert not jnp.array_equal(model.disp_arr, data.disp_arr)
    ## check that value was correct added
    t_idx = 8 ## target
    s_idx = 2 ## sample
    ## intercept only so don't need to consider x_arr
    correct_disp = data.disp_arr[t_idx, :] + 0.1
    assert model.disp_arr[t_idx, :] == correct_disp