import jax.numpy as jnp
import numpy as np
import yaml
import polars as pl
from data_module.helpers import process_bin

## test logic of bin file processing in score_test_job.py
def test_process_bin():
    with open('tests/data/splash_data_dummy.yaml', 'r') as file:
        d_dict = yaml.safe_load(file)

    start_idx = 0
    end_idx = 3
    anch_list = '/home/arosenga/score-test/tests/data/test_anch_list.csv'

    ## only read [start_idx, end_idx) rows of list
    ## +1 to start_idx because has_header = False but there _is_ a header (skip_rows will read header on starting line)
    anch_subset = pl.read_csv(anch_list, skip_rows = start_idx + 1, n_rows = end_idx - start_idx, has_header = False, separator = ',', new_columns = ['anchor', 'bin', 'first_idx', 'last_idx'])

    assert list(anch_subset.get_column('anchor')) == [
        'AAAAAAATTCAAGCGCCCGGCGGAAATTTCA', 
        'AAAAAACACTGCTGGCCAGCGCCATCGACGG', 
        'AAAAAACATCGGCGCCTTGCCGGTCTACTTC'
    ]
    assert list(anch_subset.get_column('bin')) == [1, 1, 2]
    assert list(anch_subset.get_column('first_idx')) == [0, 21, 0]
    assert list(anch_subset.get_column('last_idx')) == [20, 222, 5]

    ## loop over unique bins
    unique_bins = list(anch_subset['bin'].unique())
    assert unique_bins == [1, 2]

    for bin in unique_bins:
        ## list of anchors in bin and (polars!) dataframe of SATC for them
        bin_anchs, bin_df = process_bin(d_dict, bin, anch_subset)

        if bin == 1:
            assert bin_anchs == ['AAAAAAATTCAAGCGCCCGGCGGAAATTTCA', 
        'AAAAAACACTGCTGGCCAGCGCCATCGACGG']
            assert bin_df.shape == (223, 4) ## should only read in 0 - 222
        else:
            assert bin_anchs == ['AAAAAACATCGGCGCCTTGCCGGTCTACTTC']
            assert bin_df.shape == (6, 4)

        ## anchors will be consecutive in original list so just need to offset by index of first anchor from this bin
        offset = anch_subset.get_column('anchor').index_of(bin_anchs[0])

        ## have to loop over job anchor list to maintain indexing arithmetic
        for idx, anchor in enumerate(bin_anchs):
            j = idx + start_idx + offset

            ## get anchor table data --- drop anchor column
            anch_table_data = bin_df.filter(pl.col('anchor') == anchor)
            anch_table_data = anch_table_data.drop('anchor')

            if anchor == 'AAAAAAATTCAAGCGCCCGGCGGAAATTTCA':
                assert j == 0
                assert anch_table_data.shape == (21, 3)
                assert anch_table_data.row(0) == (0, 'GCCAAAAATCGGTGCAAAAGGGTTTTGACCG', 1)
                assert anch_table_data.row(-1) == (535, 'TCGAAAAAACGGTTCAAAAGGGTTTTGACCG', 1)
                unique_targs = len(anch_table_data.get_column('target').unique())
                unique_samps = len(anch_table_data.get_column('sample').unique())
                assert unique_targs == 19
                assert unique_samps == 14
            elif anchor == 'AAAAAACACTGCTGGCCAGCGCCATCGACGG':
                assert j == 1
                assert anch_table_data.shape == (202, 3)
                assert anch_table_data.row(0) == (105, 'AGTTTACCGCGAGATCACGAACCGCTCCGCC', 1)
                assert anch_table_data.row(-1) == (60, 'TGTTTCCCGCGCGATCACGAACCGCGCCGCC', 1)
            else:
                assert j == 2
                assert anch_table_data.shape == (6, 3)
                assert anch_table_data.row(0) == (69, 'TTGAAGGCCGCGGAGGTCTTCGAGACGCATT', 1)
                assert anch_table_data.row(-1) == (321, 'TTGATGGCCGCGGAGGTCTTCGAGACGGATT', 9)
                ## drop rows with max count < 5
                anch_table_data = anch_table_data.filter(
                    pl.col('count').max().over('target') > 5
                )
                assert anch_table_data.shape == (2, 3)

