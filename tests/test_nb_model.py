import jax.numpy as jnp
import numpy as np

def test_nb_simul(simul_data, nb_model):
    ## n_samps: 50
    ## n_targs: 100
    data = simul_data('tests/data/nb_data.yaml')
    data.resample()
    model = nb_model('tests/data/nb_model.yaml', data)
    ## data dimensions
    assert model.fx_effs_arr.shape == (100, 1)
    assert model.rd_effs_arr.shape == (100, 1)
    ## data values
    assert np.array_equal(model.rd_effs_arr, np.zeros((100, 1)))
    assert np.array_equal(model.var_comp, np.zeros((1,)))
    ## check disp_arr is not global
    all_same_disp = model.disp_arr[0, 0] * np.ones(model.disp_arr.shape)
    assert not np.array_equal(model.disp_arr, all_same_disp)
    ## check offsets
    #assert np.allclose(model.mu_arr, model.compute_mu())
    ## update fixed effects and check update_params()
    last_mu = model.compute_mu()
    new_fx_effs = model.fx_effs_arr + 1
    model.set_fx_effs_arr(new_fx_effs)
    assert np.array_equal(model.fx_effs_arr, new_fx_effs)
    #assert np.allclose(model.mu_arr, model.compute_mu())
    assert not np.allclose(model.compute_mu(), last_mu)

def test_nb_use_true(simul_data, nb_model):
    ## n_samps: 50
    ## n_targs: 100
    data = simul_data('tests/data/nb_data.yaml')
    data.resample()
    model = nb_model('tests/data/nb_model_use_true.yaml', data)
    ## data dimensions
    assert model.fx_effs_arr.shape == (100, 1)
    assert model.rd_effs_arr.shape == (100, 1)
    ## data values
    assert np.array_equal(model.rd_effs_arr, np.zeros((100, 1)))
    assert np.array_equal(model.var_comp, np.zeros((1,)))
    assert np.array_equal(model.disp_arr, data.disp_arr)
    ## check offsets
    #assert np.array_equal(model.offsets, np.zeros((data.n_samps,)))
    #assert np.array_equal(model.mu_arr, data.compute_mu())


def test_splash_use_true(splash_data, nb_model):
    data = splash_data('tests/data/splash_data_dummy.yaml', 'test_anchor')
    model = nb_model('tests/data/nb_model.yaml', data)
    ## data dimensions
    assert model.fx_effs_arr.shape == (5, 3)
    assert model.rd_effs_arr.shape == (5, 1)
    ## data values
    assert np.array_equal(model.rd_effs_arr, np.zeros((5, 1)))
    assert np.array_equal(model.var_comp, np.zeros((1,)))
    assert np.allclose(model.compute_mu(), model.mu_arr)