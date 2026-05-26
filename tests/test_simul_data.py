import jax.numpy as jnp

def test_nb_fx_targ_int_rd_slope(simul_data):
    data = simul_data('tests/configs/simul_data.yaml')
    ## check dimensions
    n_samps = data.n_samps
    n_targs = data.n_targs
    ## data dimensions
    assert data.x_arr.shape == (n_samps, 1)
    assert data.z_arr.shape == (n_samps, 1)
    assert data.fx_effs_arr.shape == (n_targs, 1)
    assert data.rd_effs_arr.shape == (n_targs, 1)
    assert data.eta_arr.shape == (n_targs, n_samps)
    assert len(data.targ_idx) == len(data.samp_idx) == len(data.counts)
    ## data values
    assert jnp.array_equal(data.x_arr, jnp.ones((n_samps, 1)))
    assert not jnp.array_equal(data.z_arr, jnp.ones((n_samps, 1)))

