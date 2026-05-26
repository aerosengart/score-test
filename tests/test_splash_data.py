import jax.numpy as jnp

## test.txt
# 76 AATTAAGCTGACCTACTGCAATGAAAG 1
# 73 AGGTGGCTGACCTGCAATGAAAGGTGC 1
# 30 AGGTGGCTGACCTGCAATGAAGGTGCA 1
# 20 AGGTGGCTGATACAGAAGGTGCATGCA 1
# 30 AGTGGCTGACCTACAATGAAAGGTGCA 1
# 73 AGTGGCTGACCTGCAAGCTGGTGCATG 1
# 76 AGTGGCTGACCTGCAATGAAAATTGCA 1
# 30 AGTGGCTGACCTGCAATGAAAGGTGCA 2

## date from clean_metadata.tsv
## 76 - 2021-12-02
## 73 - 2021-12-02
## 30 - 2021-12-02
## 20 - 2021-12-01

def test_splash_fx_targ_int_rd_slope(splash_data):
    data = splash_data('tests/configs/splash_data.yaml', 'test')

    ## check dimensions
    n_samps = data.n_samps
    n_targs = data.n_targs
    assert n_samps == 4
    assert n_targs == 8
    ## data dimensions
    assert data.x_arr.shape == (n_samps, 1)
    assert data.z_arr.shape == (n_samps, 1)
    assert len(data.targ_idx) == len(data.samp_idx) == len(data.counts)
    ## data values
    assert jnp.array_equal(data.x_arr, jnp.ones((n_samps, 1)))
    z_arr_by_hand = jnp.ones((n_samps, 1))
    z_arr_by_hand = z_arr_by_hand.at[-1, 0].set(0)
    assert jnp.array_equal(data.z_arr, z_arr_by_hand)
