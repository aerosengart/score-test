import jax.numpy as jnp

def test_nb_fx_targ_int_rd_slope(simul_data):
    data = simul_data('tests/data/nb_data.yaml')
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


def test_nb_fx_targ_int_slope_rd_slope(simul_data):
    data = simul_data('tests/data/nb_data_fx_slope.yaml')
    ## check dimensions
    n_samps = data.n_samps
    n_targs = data.n_targs
    ## data dimensions
    assert data.x_arr.shape == (n_samps, 2)
    assert data.z_arr.shape == (n_samps, 1)
    assert data.fx_effs_arr.shape == (n_targs, 2)
    assert data.rd_effs_arr.shape == (n_targs, 1)
    assert data.eta_arr.shape == (n_targs, n_samps)
    assert len(data.targ_idx) == len(data.samp_idx) == len(data.counts)
    ## data values
    assert jnp.array_equal(data.x_arr[:, 0], jnp.ones((n_samps,)))
    assert not jnp.array_equal(data.z_arr, jnp.ones((n_samps, 1)))
    ## chekc all fixed effect covariates are in [0, 2]
    assert jnp.sum(data.x_arr[:, 1] < 0) == 0
    assert jnp.sum(data.x_arr[:, 1] > 2) == 0


def test_poi_fx_targ_int_rd_slope(simul_data):
    data = simul_data('tests/data/poi_data.yaml')
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


def test_poi_fx_targ_int_batch(simul_data):
    data = simul_data('tests/data/poi_batch.yaml')
    ## check dimensions
    n_samps = data.n_samps
    n_targs = data.n_targs
    ## data dimensions
    assert data.x_arr.shape == (n_samps, 2)
    assert data.z_arr.shape == (n_samps, 1)
    assert data.fx_effs_arr.shape == (n_targs, 2)
    assert data.rd_effs_arr.shape == (n_targs, 1)
    assert data.eta_arr.shape == (n_targs, n_samps)
    assert len(data.targ_idx) == len(data.samp_idx) == len(data.counts)
    ## data values
    assert jnp.array_equal(data.x_arr[:, 0], jnp.ones((n_samps,)))
    unique_vals = jnp.unique(data.x_arr).tolist()
    wrong_vals = [x for x in unique_vals if x not in [0.0, 1.0]]
    assert len(wrong_vals) == 0
    assert not jnp.array_equal(data.z_arr, jnp.ones((n_samps, 1)))




