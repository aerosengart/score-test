import jax.numpy as jnp

## test_anchor.txt
# 58 CGGTGGGAGCCAGGGACAAGGCTCTCCATCT 1
# 58 CTCGTTGGAAACTTGAGACAAGGCTCTCCAT 1
# 108 TCGTTGAAACCAGGGACAAGGCTCCATCTTA 1
# 1 TCGTTGGAAACCAGGGACAGGAAACTATGCC 1
# 70 TCTCACGAAACCAGGGACAGGCTCTCCATCT 1
# 58 TCTCCTGTTGAAACCAGGGACGGCTCTCCAT 1
# 16 TCTCGGTGAACAAGGACGGACTCTTTCATCT 1
# 22 TCTCGGTGGGAAACCAGGGACAAGGCTCTCC 1
# 22 TCTCGTTGAAACCAAGGGACAAGGCTCTCCA 1

## date and geo_loc from clean_metadata.tsv
## 58  - 2021-12-01 - South Africa: Kwazulu-Natal
## 108 - 2021-12-02 - South Africa: Eastern Cape
## 1   - 2021-12-03 - South Africa: Eastern Cape
## 70  - 2021-12-02 - South Africa: Kwazulu-Natal
## 16  - 2021-12-03 - South Africa: Eastern Cape
## 22  - 2021-12-01 - South Africa: Kwazulu-Natal

def test_splash_fx_targ_int_rd_slope(splash_data):
    data = splash_data('tests/data/splash_data.yaml', 'test_anchor')

    ## check dimensions
    n_samps = data.n_samps
    n_targs = data.n_targs
    assert n_samps == 6
    assert n_targs == 9
    ## data dimensions
    assert data.x_arr.shape == (n_samps, 2)
    assert data.z_arr.shape == (n_samps, 1)
    assert len(data.targ_idx) == len(data.samp_idx) == len(data.counts)
    ## data values
    assert jnp.array_equal(data.x_arr[:, 0], jnp.ones((n_samps,)))
    x_arr_by_hand = jnp.array([1, 0, 0, 1, 0, 1])
    z_arr_by_hand = jnp.array([0, 1, 2, 1, 2, 0])
    assert jnp.array_equal(data.x_arr[:, 1], x_arr_by_hand)
    assert jnp.array_equal(data.z_arr[:, 0], z_arr_by_hand)
