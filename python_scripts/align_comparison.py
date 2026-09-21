#!/usr/bin/env python

"""
align_comparison.py - compare SPLASH/score test alignment results
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
parser.add_argument("--splash_file", type = str, help = "path to splash_alignment_results.tsv")
parser.add_argument("--score_file", type = str, help = "path to score_alignment_results.tsv")
parser.add_argument("--align_dir", type = str, help = "path of directory with anchor .json files")
args = parser.parse_args()

spl_df = pd.read_csv(args.splash_file, sep = '\t')
sc_df = pd.read_csv(args.score_file, sep = '\t')

spl_df['key'] = spl_df['anchor'] + '_' + spl_df['target']
sc_df['key']  = sc_df['anchor'] + '_' + sc_df['target']
dif = pd.merge(spl_df['key'], sc_df['key'], how = 'outer', indicator = True)

## overlap
ol = pd.merge(spl_df['key'], sc_df['key'], how = 'inner')
spl_ol = spl_df[spl_df['key'].isin(dif[dif['_merge'] == 'both']['key'].to_list())]
sc_ol  = sc_df[sc_df['key'].isin(dif[dif['_merge'] == 'both']['key'].to_list())]
## only select columns that have info about score test (other info is repeated)
sc_ol = sc_ol[['key', 'one_sided_test_stat', 'one_sided_p_val', 'one_sided_p_val_by', 'two_sided_test_stat', 'two_sided_p_val', 'two_sided_p_val_by']]
ol_df = pd.concat([spl_ol, sc_ol])
ol_df = ol_df.drop(columns = ['key'])
ol_df.to_csv(os.path.join(args.align_dir, 'splash_score_overlap_align.tsv'), sep = '\t')

## difference
spl_dif = spl_df[spl_df['key'].isin(dif[dif['_merge'] == 'left_only']['key'].to_list())]
sc_dif  = sc_df[sc_df['key'].isin(dif[dif['_merge'] == 'right_only']['key'].to_list())]
spl_dif = spl_dif.drop(columns = ['key'])
sc_dif = sc_dif.drop(columns = ['key'])
spl_dif.to_csv(os.path.join(args.align_dir, 'splash_not_score_align.tsv'), sep = '\t')
sc_dif.to_csv(os.path.join(args.align_dir, 'score_not_splash_align.tsv'), sep = '\t')