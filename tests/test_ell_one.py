import jax.numpy as jnp

def test_ell_one(simul_data):
    ## n_samps: 50
    ## n_targs: 100
    data = simul_data('tests/data/ell_one.yaml')
    data.resample()
    old_mix_dist = data.mix_dist
    eta_arr = data.compute_eta()
    old_counts = data.counts
    ## data dimensions
    assert data.x_arr.shape == (50, 1)
    assert data.z_arr.shape == (50, 1)
    assert eta_arr.shape == (100, 50)
    assert len(data.targ_idx) == len(data.samp_idx) == len(data.counts)
    ## data values
    assert jnp.array_equal(data.x_arr, jnp.ones((50, 1)))
    ## resample and check things didn't change
    data.resample()
    eta_arr = data.compute_eta()
    ## data dimensions
    assert data.x_arr.shape == (50, 1)
    assert data.z_arr.shape == (50, 1)
    assert eta_arr.shape == (100, 50)
    assert len(data.targ_idx) == len(data.samp_idx) == len(data.counts)
    ## data values
    assert jnp.array_equal(data.x_arr, jnp.ones((50, 1)))
    assert data.counts != old_counts
    assert jnp.array_equal(data.mix_dist, old_mix_dist)

