import jax.numpy as jnp
import numpy as np

def test_poi_fx_targ_int_rd_slope(simul_data):
    ## n_samps: 50
    ## n_targs: 100
    data = simul_data('tests/data/poi_data.yaml')
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
    assert np.array_equal(data.x_arr, np.ones((50, 1)))
    assert not np.array_equal(data.z_arr, np.ones((100, 1)))


def test_poi_fx_targ_int_batch(simul_data):
    ## n_samps: 100
    ## n_targs: 100
    data = simul_data('tests/data/poi_batch.yaml')
    data.resample()
    eta_arr = data.compute_eta()
    ## data dimensions
    assert data.x_arr.shape == (100, 2)
    assert data.z_arr.shape == (100, 1)
    assert data.fx_effs_arr.shape == (100, 2)
    assert data.rd_effs_arr.shape == (100, 1)
    assert eta_arr.shape == (100, 100)
    assert len(data.targ_idx) == len(data.samp_idx) == len(data.counts)
    ## data values
    assert np.array_equal(data.x_arr[:, 0], np.ones((100,)))
    unique_vals = np.unique(data.x_arr).tolist()
    wrong_vals = [x for x in unique_vals if x not in [0.0, 1.0]]
    assert len(wrong_vals) == 0
    assert not np.array_equal(data.z_arr, np.ones((100, 1)))


def test_poi_batch_z(simul_data):
    ## n_samps: 50
    ## n_targs: 100
    ## batch_3
    data = simul_data('tests/data/poi_batch_z.yaml')
    data.resample()
    eta_arr = data.compute_eta()
    ## data dimensions
    assert data.x_arr.shape == (50, 1)
    assert data.z_arr.shape == (50, 2)
    assert data.fx_effs_arr.shape == (100, 1)
    assert data.rd_effs_arr.shape == (100, 2)
    assert eta_arr.shape == (100, 50)
    assert len(data.targ_idx) == len(data.samp_idx) == len(data.counts)
    ## data values
    assert np.array_equal(data.x_arr[:, 0], np.ones((50,)))
    ## check all fixed effect covariates are in [0, 1]
    assert np.sum(data.z_arr[:, 1:] < 0) == 0
    assert np.sum(data.z_arr[:, 1:] > 2) == 0