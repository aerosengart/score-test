import jax.numpy as jnp

def test_nb_simul(simul_data, nb_model):
    data = simul_data('tests/data/nb_data.yaml')
    model = nb_model('tests/data/nb_model_use_true.yaml', data)

    ## check dimensions
    n_samps = data.n_samps
    n_targs = data.n_targs
    ## data dimensions
    assert model.fx_effs_arr.shape == (n_targs, 1)
    assert model.rd_effs_arr.shape == (n_targs, 1)
    ## data values
    assert jnp.array_equal(model.rd_effs_arr, jnp.zeros((n_targs, 1)))
    assert jnp.array_equal(model.tau_sqr, jnp.zeros((data.q,)))
    assert jnp.array_equal(model.gamma, data.gamma)
    ## check offsets
    assert jnp.array_equal(model.offsets, data.offsets)
    assert jnp.array_equal(model.mu_arr, data.mu_arr)