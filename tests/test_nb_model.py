import jax.numpy as jnp

def test_nb_simul(simul_data, nb_model):
    data = simul_data('tests/configs/simul_data.yaml')
    model = nb_model('tests/configs/nb_model.yaml', data)

    ## check dimensions
    n_samps = data.n_samps
    n_targs = data.n_targs
    ## data dimensions
    assert model.fx_effs_arr.shape == (n_targs, 1)
    assert model.rd_effs_arr.shape == (n_targs, 1)
    ## data values
    assert jnp.array_equal(model.rd_effs_arr, jnp.zeros((n_targs, 1)))
    assert jnp.array_equal(model.tau_sqr, jnp.zeros((data.q,)))
