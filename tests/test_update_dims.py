import jax.numpy as jnp
import numpy as np

def test_nb_update_n_targs_smaller(simul_data):
    ## 3 level batch effect
    data = simul_data('tests/data/nb_data_fx_slope.yaml')
    data.resample()
    
    old_fx_effs = data.fx_effs_arr
    old_rd_effs = data.rd_effs_arr
    old_disp    = data.disp_arr
    old_x       = data.x_arr
    old_z       = data.z_arr
    old_targ_idx = data.targ_idx
    old_samp_idx = data.samp_idx
    old_counts   = data.counts
    old_y = data.construct_y()
    ## do update
    data.update_n_targs(10)
    ## check dimensions
    p = data.fx_effs_arr.shape[-1]
    q = data.rd_effs_arr.shape[-1]
    assert data.n_targs == 10
    assert data.fx_effs_arr.shape == (10, p)
    assert data.rd_effs_arr.shape == (10, q)
    assert data.disp_arr.shape == (10, 1)
    assert data.x_arr.shape == (data.n_samps, p)
    assert data.z_arr.shape == (data.n_samps, q)
    assert len(data.targ_idx) == len(data.samp_idx)
    assert len(data.targ_idx) == len(data.counts)
    ## check that we kept the first ten targets
    assert np.array_equal(data.fx_effs_arr, old_fx_effs[0:10, :])
    assert np.array_equal(data.rd_effs_arr, old_rd_effs[0:10, :])
    assert np.array_equal(data.disp_arr, old_disp[0:10, :])
    assert np.array_equal(data.x_arr, old_x) ## shouldn't change
    assert np.array_equal(data.z_arr, old_z) ## shouldn't change
    assert max(data.targ_idx) <= 10
    ## check counts
    assert old_targ_idx[:len(data.targ_idx)] == data.targ_idx
    assert old_samp_idx[:len(data.targ_idx)] == data.samp_idx
    assert old_counts[:len(data.targ_idx)]   == data.counts
    assert np.array_equal(old_y[:10, :], data.construct_y())



# def test_nb_update_n_targs_bigger(simul_data):
#     ## 3 level batch effect
#     data = simul_data('tests/data/nb_data_fx_slope.yaml')
#     old_fx_effs = data.fx_effs_arr
#     old_rd_effs = data.rd_effs_arr
#     old_disp    = data.disp_arr
#     old_x       = data.x_arr
#     old_z       = data.z_arr
#     old_targ_idx = data.targ_idx
#     old_samp_idx = data.samp_idx
#     old_counts   = data.counts
#     old_col_sums = data.col_sums
#     old_y       = data.construct_y()
#     ## do update
#     data.update_n_targs(101)
#     ## check dimensions
#     p = data.fx_effs_arr.shape[-1]
#     q = data.rd_effs_arr.shape[-1]
#     assert data.n_targs == 101
#     assert data.fx_effs_arr.shape == (101, p)
#     assert data.rd_effs_arr.shape == (101, q)
#     assert data.disp_arr.shape == (101, 1)
#     assert data.x_arr.shape == (data.n_samps, p)
#     assert data.z_arr.shape == (data.n_samps, q)
#     assert len(data.targ_idx) == len(data.samp_idx)
#     assert len(data.targ_idx) == len(data.counts)
#     ## check that we kept the first ten targets
#     assert np.array_equal(data.fx_effs_arr[0:100, :], old_fx_effs)
#     assert np.array_equal(data.rd_effs_arr[0:100, :], old_rd_effs)
#     assert np.array_equal(data.disp_arr[0:100, :], old_disp)
#     assert np.array_equal(data.x_arr, old_x)
#     assert np.array_equal(data.z_arr, old_z)
#     assert max(data.targ_idx) <= 101
#     ## check counts
#     assert np.array_equal(old_col_sums, data.col_sums) ## shouldn't change
#     assert data.targ_idx[:len(old_targ_idx)] == old_targ_idx
#     assert data.samp_idx[:len(old_targ_idx)] == old_samp_idx
#     assert data.counts[:len(old_targ_idx)]   == old_counts
#     assert np.array_equal(old_y, data.construct_y()[:100, :])


def test_nb_update_n_samps_smaller(simul_data):
    ## 3 level batch effect
    data = simul_data('tests/data/nb_data_fx_slope.yaml')
    data.resample()
    old_fx_effs = data.fx_effs_arr
    old_rd_effs = data.rd_effs_arr
    old_disp    = data.disp_arr
    old_y       = data.construct_y()
    old_x, old_z, old_x_names, old_z_names = data.get_covars(ctr = False)
    ## do update
    data.update_n_samps(10)
    ## check dimensions
    p = data.fx_effs_arr.shape[-1]
    q = data.rd_effs_arr.shape[-1]
    assert data.n_samps == 10
    assert data.fx_effs_arr.shape == (data.n_targs, p)
    assert data.rd_effs_arr.shape == (data.n_targs, q)
    assert data.disp_arr.shape == (data.n_targs, 1)
    assert data.x_arr.shape == (10, p)
    assert data.z_arr.shape == (10, q)
    assert len(data.targ_idx) == len(data.samp_idx)
    assert len(data.targ_idx) == len(data.counts)
    ## check that we kept the first ten targets (compare uncentered values)
    new_x, new_z, new_x_names, new_z_names = data.get_covars(ctr = False)
    assert np.allclose(new_x, old_x[:10, :])
    assert np.allclose(new_z, old_z[:10, :])
    assert np.array_equal(data.disp_arr, old_disp)
    assert np.array_equal(data.fx_effs_arr, old_fx_effs)
    assert np.array_equal(data.rd_effs_arr, old_rd_effs)
    assert max(data.samp_idx) <= 10
    ## check counts 
    assert np.array_equal(old_y[:, :10], data.construct_y())
    ## check column names
    assert new_x_names == old_x_names
    assert new_z_names == old_z_names

# def test_nb_update_n_samps_bigger(simul_data):
#     ## 3 level batch effect
#     data = simul_data('tests/data/nb_data_fx_slope.yaml')
#     data.resample()
#     old_fx_effs = data.fx_effs_arr
#     old_rd_effs = data.rd_effs_arr
#     old_disp    = data.disp_arr
#     old_n_samps = data.n_samps
#     old_col_sums = data.col_sums
#     old_targ_idx = data.targ_idx
#     old_samp_idx = data.samp_idx
#     old_counts   = data.counts
#     old_y = data.construct_y()
#     old_x, old_z = data.get_covars(ctr = False)
#     ## do update
#     data.update_n_samps(51)
#     ## check dimensions
#     p = data.fx_effs_arr.shape[-1]
#     q = data.rd_effs_arr.shape[-1]
#     assert data.n_samps == 51
#     assert data.fx_effs_arr.shape == (data.n_targs, p)
#     assert data.rd_effs_arr.shape == (data.n_targs, q)
#     assert data.disp_arr.shape == (data.n_targs, 1)
#     assert len(data.col_sums) == 51
#     assert data.x_arr.shape == (51, p)
#     assert data.z_arr.shape == (51, q)
#     assert len(data.targ_idx) == len(data.samp_idx)
#     assert len(data.targ_idx) == len(data.counts)
#     ## check that we kept the first targets --- compare uncentered values
#     new_x, new_z = data.get_covars(ctr = False)
#     assert np.allclose(new_x[:old_n_samps, :], old_x)
#     assert np.allclose(new_z[:old_n_samps, :], old_z)
#     assert np.array_equal(data.col_sums[:old_n_samps], old_col_sums)
#     assert np.array_equal(data.disp_arr, old_disp)
#     assert np.array_equal(data.fx_effs_arr, old_fx_effs)
#     assert np.array_equal(data.rd_effs_arr, old_rd_effs)
#     assert max(data.samp_idx) <= 51
#     ## check counts
#     assert np.array_equal(data.col_sums[:old_n_samps], old_col_sums)
#     # assert data.targ_idx[:len(old_samp_idx)] == old_targ_idx
#     # assert data.samp_idx[:len(old_samp_idx)] == old_samp_idx
#     # assert data.counts[:len(old_samp_idx)]   == old_counts
#     assert np.array_equal(old_y, data.construct_y()[:, :50])
