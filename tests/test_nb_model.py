import jax.numpy as jnp

def test_nb_simul(simul_data, nb_model):
    data = simul_data('tests/data/nb_data.yaml')
    model = nb_model('tests/data/nb_model.yaml', data)

    ## check dimensions
    n_samps = data.n_samps
    n_targs = data.n_targs
    ## data dimensions
    assert model.fx_effs_arr.shape == (n_targs, 1)
    assert model.rd_effs_arr.shape == (n_targs, 1)
    ## data values
    assert jnp.array_equal(model.rd_effs_arr, jnp.zeros((n_targs, 1)))
    assert jnp.array_equal(model.var_comp, jnp.zeros((data.q,)))
    ## check disp_arr is not global
    assert not jnp.array_equal(model.disp_arr, model.disp_arr[0, 0] * jnp.ones(model.disp_arr.shape))
    ## check offsets
    assert jnp.allclose(model.mu_arr, model.compute_mu())
    ## update fixed effects and check update_params()
    last_mu = model.mu_arr
    new_fx_effs = model.fx_effs_arr + 1
    model.set_fx_effs_arr(new_fx_effs)
    model.update_params()
    assert jnp.array_equal(model.fx_effs_arr, new_fx_effs)
    assert jnp.allclose(model.mu_arr, model.compute_mu())
    assert not jnp.allclose(model.mu_arr, last_mu)
