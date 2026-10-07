#!/usr/bin/env python

"""
signf_anchors.py - Create lists of significant anchors (level = 0.05) according to SPLASH and score test method.
ARGS: 
    splash_file     - str; path to result.after_correction.all_anchors.tsv file
    score_file      - str; path to anchor_score_test_summary.csv file
    output          - str; path to output directory
RAISES:  None
RETURNS: None
"""

import os
import sys
import jsonargparse
import yaml
import json

import pandas as pd
import numpy as np

from tqdm import tqdm

sys.path.append('/home/arosenga/score-test')
from data_module.dispatch  import data_types



## create an argument parser and add options
parser = jsonargparse.ArgumentParser()
parser.add_argument('--cfg', action = jsonargparse.ActionConfigFile)
parser.add_argument('--d_config', 
                    help = 'Path to data config file.', 
                    dest = 'd_config',
                    type = jsonargparse.typing.Path)
parser.add_argument("--splash_file", type = str, help = "path to result.after_correction.all_anchors.tsv file")
parser.add_argument("--score_file", type = str, help = "path to anchor_score_test_summary.csv file")
parser.add_argument("--alpha", type = float, help = "level cut-off")
parser.add_argument("--output", type = str, help = "path to output directory")
args = parser.parse_args()

## check if save path exists
out_path = os.path.join(args.output)
if not os.path.isdir(out_path):
    os.makedirs(out_path)

## basic splash data dict so we can construct the target-sample table
with open(args.d_config, 'r') as file:
    d_dict = yaml.safe_load(file)

## read in files and sort by descending total counts
splash = pd.read_csv(args.splash_file, sep = '\t', header = 0)
splash = splash.sort_values(by = 'M', ascending = False)
score  = pd.read_csv(args.score_file, sep = ',', header = 0)
score  = score.sort_values(by = 'M', ascending = False)

## restrict score results to only the anchors that were used in the score test
splash = splash[splash['anchor'].isin(score['anchor'])]

## filter splash to signf/insignf anchors with large effect size
splash_signf   = splash[splash['pval_opt_corrected'] <= args.alpha]
splash_insignf = splash[splash['pval_opt_corrected'] > args.alpha]
splash_signf   = splash_signf[splash_signf['effect_size_bin'] > 0.5]
## filter score file to signf anchors
score_signf   = score[score['one_sided_p_val_by'] <= args.alpha]
score_insignf = score[score['one_sided_p_val_by'] > args.alpha]

## filter to M >= 30
splash_signf = splash_signf[splash_signf['M'] >= 30]
score_signf = score_signf[score_signf['M'] >= 30]

## filter to n_samps > 1 and n_targs > 1 --- i think this is already the case
splash_signf = splash_signf[splash_signf['number_nonzero_samples'] > 1]
score_signf = score_signf[score_signf['n_samps'] > 1]
splash_signf = splash_signf[splash_signf['anch_uniqTargs'] > 1]
score_signf = score_signf[score_signf['n_targs'] > 1]

## get controls
splash_n = splash_signf.shape[0]
score_n  = score_signf.shape[0]
splash_insignf = splash_insignf[:splash_n] 
score_insignf  = score_insignf[:score_n]

## create lists and write to disc
splash_calls = pd.concat([splash_signf, splash_insignf])
score_calls  = pd.concat([score_signf, score_insignf])
splash_calls.to_csv(os.path.join(out_path, 'splash_calls.csv'), index = False)
score_calls.to_csv(os.path.join(out_path, 'score_calls.csv'), index = False)

## initialize filtered call dfs
splash_file = os.path.join(out_path, 'splash_calls_filt.csv')
score_file = os.path.join(out_path, 'score_calls_filt.csv')
with open(splash_file, 'w') as file:
    file.write('anchor,target,targ_counts,nonzero_samps\n')
with open(score_file, 'w') as file:
    file.write('anch_idx,anchor,target,targ_counts,nonzero_samps\n')

## loop over rows
print('Looping over SPLASH calls...')
for i in tqdm(range(len(splash_calls.index))):
    row = splash_calls.iloc[i, :]
    anchor = row['anchor']
    ## construct target-sample table and filter again
    data  = data_types[d_dict['outcome']](d_dict, anchor)
    y_table = data.construct_y()
    m = y_table.sum()
    ## remove samples with 5 or fewer counts
    col_sums = y_table.sum(axis = 0)
    col_sums_g5 = [i for i, x in enumerate(col_sums) if x > 5]
    y_table = y_table[:, col_sums_g5]
    ## target counts > 5% of total
    row_sums = y_table.sum(axis = 1)
    row_sums_g5p = [i for i, x in enumerate(row_sums) if x > 0.05 * m]
    ## loop over called targets
    for j in range(len(row_sums_g5p)):
        row_data = f'{str(anchor)},{str(data.targets[row_sums_g5p[j]])},{row_sums[row_sums_g5p[j]]},{str(y_table[row_sums_g5p[j], :].astype(bool).sum())}\n'
        with open(splash_file, 'a') as file:
            file.write(row_data)

## loop over rows
print('Looping over score test calls...')
for i in tqdm(range(len(score_calls.index))):
    row = score_calls.iloc[i, :]
    anchor = row['anchor']
    ## construct target-sample table and filter again
    data  = data_types[d_dict['outcome']](d_dict, anchor)
    y_table = data.construct_y()
    m = y_table.sum()
    ## remove samples with 5 or fewer counts
    col_sums = y_table.sum(axis = 0)
    col_sums_g5 = [i for i, x in enumerate(col_sums) if x > 5]
    y_table = y_table[:, col_sums_g5]
    ## target counts > 5% of total
    row_sums = y_table.sum(axis = 1)
    row_sums_g5p = [i for i, x in enumerate(row_sums) if x > 0.05 * m]
    ## loop over called targets
    for j in range(len(row_sums_g5p)):
        row_data = f'{str(row['idx'])},{str(anchor)},{str(data.targets[row_sums_g5p[j]])},{row_sums[row_sums_g5p[j]]},{str(y_table[row_sums_g5p[j], :].astype(bool).sum())}\n'
        with open(score_file, 'a') as file:
            file.write(row_data)
    