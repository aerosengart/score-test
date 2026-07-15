import jax.numpy as jnp

def test_alt_splice(simul_data):
    ## n_samps: 50
    ## n_targs: 100
    data = simul_data('tests/data/alt_splice.yaml')
    data.resample()
    eta_arr = data.compute_eta()
    ## data dimensions
    assert data.x_arr.shape == (50, 1)
    assert data.z_arr.shape == (50, 1)
    assert eta_arr.shape == (100, 50)
    assert len(data.targ_idx) == len(data.samp_idx) == len(data.counts)
    ## data values
    assert jnp.array_equal(data.x_arr, jnp.ones((50, 1)))
    ## first half are -1, second half are 1
    assert jnp.array_equal(data.z_arr[:25, 1], -jnp.ones((25,)))
    assert jnp.array_equal(data.z_arr[25:, 1], jnp.ones((25,)))

