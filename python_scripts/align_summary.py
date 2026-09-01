#!/usr/bin/env python

"""
align_summary.py - summarize filtered alignment output 
REQUIRES: 
    splash_file - path to splash_calls.csv
    score_file  - path to score_calls.csv
    align_dir   - path to output of alignment step
MODIFIES: None
RETURNS: None
"""

import os
import json
import argparse

import pandas as pd

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--splash_file", type = str, help = "path to splash_calls.csv")
parser.add_argument("--score_file", type = str, help = "path to score_calls.csv")
parser.add_argument("--align_dir", type = str, help = "path of directory with anchor .json files")
args = parser.parse_args()

## read in specific columns (no entropy cols or targets > 5th rank)
spl_cols = list(pd.read_csv(args.splash_file, nrows = 1))
use_cols = [x for x in spl_cols if 'seq_entropy' not in x]
use_cols = [x for x in use_cols if not any(i in x for i in ['_6', '_7', '_8', '_9', '_10'])]
spl_res = pd.read_csv(args.splash_file, usecols = use_cols)

sc_cols = list(pd.read_csv(args.score_file, nrows = 1))
use_cols = [x for x in sc_cols if 'seq_entropy' not in x]
use_cols = [x for x in use_cols if not any(i in x for i in ['_6', '_7', '_8', '_9', '_10'])]
sc_res = pd.read_csv(args.score_file, usecols = use_cols)

## rename columns so that stub is first
spl_res.columns = [*spl_res.columns.str.replace('cnt_most_freq_target', 'target_cnt')]
spl_res.columns = [*spl_res.columns.str.replace('most_freq_target', 'target_seq')]
sc_res.columns = [*sc_res.columns.str.replace('cnt_most_freq_target', 'target_cnt')]
sc_res.columns = [*sc_res.columns.str.replace('most_freq_target', 'target_seq')]

## pivot longer (abundant targs are columns but we need to be able to merge dfs)
spl_long = pd.wide_to_long(spl_res, stubnames = ['target_cnt', 'target_seq'], i = 'anchor', j = 'target_rank', sep = '_')
sc_long = pd.wide_to_long(sc_res, stubnames = ['target_cnt', 'target_seq'], i = 'anchor', j = 'target_rank', sep = '_')

## read in alignment results for each test
spl_aligns = pd.read_csv(f'{args.align_dir}/splash_strain_def.tsv', sep = '\t', header = 0)
sc_aligns  = pd.read_csv(f'{args.align_dir}/score_strain_def.tsv', sep = '\t', header = 0)

## inner join on anchor/target pairs
spl_join = pd.merge(
    left = spl_long, right = spl_aligns, how = 'inner', left_on = ['anchor', 'target_seq'], right_on = ['anchor', 'target']
)
sc_join  = pd.merge(
    left = sc_long, right = sc_aligns, how = 'inner', left_on = ['anchor', 'target_seq'], right_on = ['anchor', 'target']
)

## drop unneeded columns
spl_final = spl_join[['anchor', 'target', 'target_cnt', 'M', 'anch_uniqTargs', 'effect_size_bin', 'nonzero_samps', 'pval_base', 'pval_opt', 'pval_asymp_opt', 'pval_opt_corrected', 'anch_align_NC_045512.2', 'anch_align_Delta', 'anch_align_Omicron_BA1', 'anch_align_Omicron_BA2', 'targ_align_NC_045512.2', 'targ_align_Delta', 'targ_align_Omicron_BA1', 'targ_align_Omicron_BA2']]
sc_final = sc_join[['anchor', 'target', 'target_cnt', 'M', 'anch_uniqTargs', 'effect_size_bin', 'nonzero_samps', 'pval_base', 'pval_opt', 'pval_asymp_opt', 'pval_opt_corrected', 'one_sided_test_stat', 'one_sided_p_val', 'one_sided_p_val_by', 'two_sided_test_stat','two_sided_p_val', 'two_sided_p_val_by', 'anch_align_NC_045512.2', 'anch_align_Delta', 'anch_align_Omicron_BA1', 'anch_align_Omicron_BA2', 'targ_align_NC_045512.2', 'targ_align_Delta', 'targ_align_Omicron_BA1', 'targ_align_Omicron_BA2']]

## sort
spl_final = spl_final.sort_values(by = ['anchor', 'target_cnt'])
sc_final  = sc_final.sort_values(by = ['anchor', 'target_cnt'])

## write to file
spl_final.to_csv(f'{args.align_dir}/splash_alignment_results.tsv', sep = '\t', index = False)
sc_final.to_csv(f'{args.align_dir}/score_alignment_results.tsv', sep = '\t', index = False)





