#!/usr/bin/env python

import argparse
import os
import math
import yaml
import jsonargparse
import subprocess
import pandas    as pd
import polars    as pl

from tqdm import tqdm

def process_bin(d_config, bin, anch_subset):
    bin_file = os.path.join(d_config['paths']['temp_dir'], f'bin{bin}.satc')
    ## dump SATC binary files as text if not already done
    if not os.path.isfile(bin_file):
        subprocess.call([
            d_config['paths']['splash_dir'] + '/satc_dump', 
            os.path.join(d_config['paths']['bin_dir'], f'bin{bin}.satc'),
            os.path.join(d_config['paths']['temp_dir'], f'bin{bin}.satc'),
        ])
    ## rows in anchor list that correspond to anchors for this job and this bin
    one_bin_file = anch_subset.filter(pl.col('bin') == bin)
    bin_anch_tab = pl.DataFrame(schema = {
        'sample': pl.Int64,
        'anchor': pl.String,
        'target': pl.String,
        'count':  pl.Int64
    })
    ## loop over anchors
    for i in range(one_bin_file.shape[0]):
        row = one_bin_file.row(i, named = True)
        start_idx = row['start_idx']
        nrow = row['end_idx'] - start_idx + 1 ## +1 to correct 0-indexing
        anch_data = pl.read_csv(bin_file, skip_rows = start_idx, n_rows = nrow, has_header = False, separator = '\t', new_columns = ['sample', 'anchor', 'target', 'count'])
        bin_anch_tab = pl.concat([bin_anch_tab, anch_data])

    return list(one_bin_file.get_column('anchor')), bin_anch_tab


################################################################################

parser = jsonargparse.ArgumentParser()
parser.add_argument('--cfg', action = jsonargparse.ActionConfigFile)
parser.add_argument('--d_config', 
                    help = 'Path to data config file.', 
                    dest = 'd_config',
                    type = jsonargparse.typing.Path)
parser.add_argument("--splash_file", type = str, help = "path to result.after_correction.all_anchors.tsv")
parser.add_argument("--splash_ids", type = str, help = 'path to sample_name_to_id_mapping.txt')
parser.add_argument("--anchor_bin_file", type = str, help = "path to anchor_bin_key.csv")
parser.add_argument("--m_thresh", type = int, help = "total anchor table count minimum")
parser.add_argument("--uniq_targs", type = int, help = "number of unique targets minimum")
parser.add_argument("--nonzero_samps", type = int, help = "number of non-zero samples minimum")
parser.add_argument("--output", type = str, help = "path to output directory")
args = parser.parse_args()

## read in files
result_df       = pl.read_csv(args.splash_file, separator = '\t', has_header = True)
anchor_bin_file = pl.read_csv(args.anchor_bin_file, separator = ',', has_header = True)

## subset
result_df = result_df.filter((pl.col('anch_uniqTargs') >= args.uniq_targs) & (pl.col('number_nonzero_samples') >= args.nonzero_samps) & (pl.col('M') >= args.m_thresh))

## filter to those anchors
anchors = result_df['anchor'].to_list()
anchor_bin_file = anchor_bin_file.filter(pl.col('anchor').is_in(anchors))

#[anchor_bin_file['anchor'].isin(anchors)]
splash_ids = pl.read_csv(args.splash_ids, separator = ' ', has_header = False, new_columns = ['ERS', 'id'])
splash_ids = splash_ids.with_columns([
    pl.col('id').cast(pl.String)
])

with open(args.d_config, 'r') as file:
    d_dict = yaml.safe_load(file)

## output file
anchor_tab_offsets_file = os.path.join(args.output, 'anchor_column_sums.csv')

if not os.path.isfile(anchor_tab_offsets_file):
    with open(anchor_tab_offsets_file, 'w') as f:
        f.write(f'anchor,{','.join(splash_ids['id'].to_list())}\n')

    ## loop over unique bins
    unique_bins = list(anchor_bin_file['bin'].unique())
    for bin in tqdm(unique_bins):
        anch_subset = anchor_bin_file.filter(pl.col('bin') == bin)
        ## list of anchors in bin and (polars!) dataframe of SATC for them
        bin_anchs, bin_df = process_bin(d_dict, bin, anch_subset)
        ## loop over all anchors in bin
        for idx, anchor in enumerate(bin_anchs):
            ## get anchor table data --- drop anchor column
            anch_table_data = bin_df.filter(pl.col('anchor') == anchor)
            anch_table_data = anch_table_data.drop('anchor')
            ## filter
            unique_targs0 = len(anch_table_data.get_column('target').unique())
            unique_samps0 = len(anch_table_data.get_column('sample').unique())
            anch_table_data = anch_table_data.filter(
                pl.col('count').max().over('target') >= 10
            )
            ## skip anchor if table is too small
            unique_targs1 = len(anch_table_data.get_column('target').unique())
            unique_samps1 = len(anch_table_data.get_column('sample').unique())
            if unique_targs1 < 30 or unique_samps1 < 200:
                continue
            ## write column sums to file
            nonzero_col_sums = anch_table_data.group_by('sample').agg(
                pl.col('count').sum().alias('col_sum1')
            )
            acs_data = [range(len(splash_ids['id'])), [0] * len(splash_ids['id'])]
            all_col_sums = pl.DataFrame(acs_data, schema = ['sample', 'col_sum2'])
            joined = all_col_sums.join(nonzero_col_sums, on = 'sample', how = 'left').with_columns(
                pl.selectors.numeric().fill_null(0)
            )
            joined = joined.with_columns(
                (pl.col('col_sum1') + pl.col('col_sum2')).alias('col_sum')
            )
            joined_sums = joined.with_columns([
                    pl.col('col_sum').cast(pl.String)
                ]).get_column('col_sum').to_list()
            with open(anchor_tab_offsets_file, 'a') as f:
                f.write(f'{anchor},{','.join(joined_sums)}\n') 

## compute offsets and save to file
col_sums_df = pl.read_csv(anchor_tab_offsets_file, separator = ',', has_header = True)
## drop anchor column
col_sums = col_sums_df.drop('anchor').sum()
col_sums = col_sums.transpose(include_header = True, header_name = 'sample', column_names = ['col_sum'])
col_sums = col_sums.with_columns(
    pl.col('col_sum').log().alias('log_col_sum')
)
denom = col_sums.select(pl.col('log_col_sum').sum()).item() / col_sums.shape[0]
col_sums = col_sums.with_columns(
    (pl.col('col_sum') / math.exp(denom)).log().alias('offset')
)
col_sums.write_csv(os.path.join(args.output, 'anchor_offsets.csv'), include_header = True)

## compute filtered anchor bin file
kept_anchors = col_sums_df.get_column('anchor').to_list()
anchor_bin_kept = anchor_bin_file.filter(pl.col('anchor').is_in(kept_anchors))
anchor_bin_kept.write_csv(os.path.join(args.output, 'anchor_bin_key_filter.csv'))





