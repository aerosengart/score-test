import jax.numpy as jnp

## test_anchor.txt
# 286	AGTTTCCCGCGCGATCACGAACCGCGCCGCA	1
# 470	AGTTTCCCGCGCGATCACGAACCGCGCCGCC	1
# 400	AGTTTCCCGCGCGATCACGAACCGCGCCGCC	1
# 87	AGTTTCCCGCGCGATCACGAACCGCGCCGCC	1
# 470	CATTTCCCGCGCGATCACGAACCGCGCCGCC	4
# 133	CCTTTCCCGCGCGATCACGAACCGCGCCGCC	1
# 400	CCTTTCCCGCGCGATCACGAACCGCGCCGCC	1
# 584	CGAGTCCCGCTCTATCACGAACCGCGCCGCC	1

## date and lab from data/amman/clean_metadata.tsv
# 87  2020-12-20 (49)  Instititut für Mikrobiologie; Uni Innsbruck (0, 0)
# 133 2021-02-21 (112) Instititut für Mikrobiologie; Uni Innsbruck (0, 0)
# 286 2021-02-07 (98)  TU Wien - Institut für Wassergüte (0, 1)
# 400 2021-01-22 (82)  Instititut für Mikrobiologie; Uni Innsbruck (0, 0)
# 470 2021-01-11 (71)  MedUni Innsbruck - Institut für Gerichtliche Medizin (1, 0)
# 584 2021-02-09 (100) TU Wien - Institut für Wassergüte (0, 1)

def test_splash_fx_targ_int_rd_slope_dummy(splash_data):
    data = splash_data('tests/data/splash_data_dummy.yaml', 'test_anchor')
    ## check dimensions
    n_samps = data.n_samps
    n_targs = data.n_targs
    assert n_samps == 6
    assert n_targs == 5
    ## data dimensions
    assert data.x_arr.shape == (n_samps, 3)
    assert data.z_arr.shape == (n_samps, 1)
    assert len(data.targ_idx) == len(data.samp_idx) == len(data.counts)
    ## data values
    assert jnp.array_equal(data.x_arr[:, 0], jnp.ones((n_samps,)))
    x_arr_by_hand = jnp.array([[0, 1], [1, 0], [0, 0], [0, 0], [0, 0], [0, 1]])
    z_arr_by_hand = jnp.array([[98], [71], [82], [49], [112], [100]])
    counts_by_hand = jnp.array([1, 1, 1, 1, 4, 1, 1, 1])
    assert jnp.array_equal(data.x_arr[:, 1:], x_arr_by_hand)
    assert jnp.array_equal(data.z_arr, z_arr_by_hand)
    assert jnp.array_equal(data.counts, counts_by_hand)
