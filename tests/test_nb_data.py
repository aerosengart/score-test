import jax.numpy as jnp

def test_nb_fx_targ_int_rd_slope(simul_data):
    ## n_samps: 50
    ## n_targs: 100
    ## no fixed slopes
    data = simul_data('tests/data/nb_data.yaml')
    data.resample()
    eta_arr = data.compute_eta()
    ## data dimensions
    assert data.x_arr.shape == (50, 1)
    assert data.z_arr.shape == (50, 1)
    assert data.fx_effs_arr.shape == (100, 1)
    assert data.rd_effs_arr.shape == (100, 1)
    assert eta_arr.shape == (100, 50)
    assert len(data.targ_idx) == len(data.samp_idx) == len(data.counts)
    ## data values
    assert jnp.array_equal(data.x_arr, jnp.ones((50, 1)))
    assert not jnp.array_equal(data.z_arr, jnp.ones((50, 1)))
    ## check that linear predictor is computed correctly
    assert jnp.allclose(eta_arr - data.fx_effs_arr @ data.x_arr.T, jnp.zeros(eta_arr.shape))
    ## dispersion is same for all targets
    assert jnp.array_equal(data.disp_arr, 5 * jnp.ones((100, 1)))


def test_nb_fx_targ_int_slope_rd_slope(simul_data):
    ## n_samps: 50
    ## n_targs: 100
    ## 3 level batch effect --- 2 dummy variables
    data = simul_data('tests/data/nb_data_fx_slope.yaml')
    data.resample()
    eta_arr = data.compute_eta()
    ## data dimensions
    assert data.x_arr.shape == (50, 3)
    assert data.z_arr.shape == (50, 1)
    assert data.fx_effs_arr.shape == (100, 3)
    assert data.rd_effs_arr.shape == (100, 1)
    assert eta_arr.shape == (100, 50)
    assert len(data.targ_idx) == len(data.samp_idx) == len(data.counts)
    ## data values
    assert jnp.array_equal(data.x_arr[:, 0], jnp.ones((50,)))
    assert not jnp.array_equal(data.z_arr, jnp.ones((50, 1)))
    ## check that linear predictor is computed correctly
    assert jnp.allclose((eta_arr - data.fx_effs_arr @ data.x_arr.T)[0, :], jnp.zeros(eta_arr.shape))
    ## check all fixed effect covariates are in [0, 1]
    assert jnp.sum(data.x_arr[:, 1:] < 0) == 0
    assert jnp.sum(data.x_arr[:, 1:] > 1) == 0
    ## dispersion is same for all targets
    assert jnp.array_equal(data.disp_arr, 5 * jnp.ones((100, 1)))


