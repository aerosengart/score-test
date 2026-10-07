#!/usr/bin/env python

"""
json_summary.py - summarize .json output of score test for anchor tables
REQUIRES: json_dir - str path; directory of anchor-specific .json output
MODIFIES: None
RETURNS: None
"""

import os
import json
import argparse

import pandas as pd

from tqdm import tqdm

from scipy.stats import false_discovery_control as fdc

parser = argparse.ArgumentParser()
parser.add_argument("--json_dir", type = str, help = "path of directory with anchor .json files")
parser.add_argument("--splash_out_dir", type = str, help = "path of directory with SPLASH output")
parser.add_argument("--out_dir", type = str, help = "output directory")
args = parser.parse_args()

out_file = os.path.join(args.out_dir, 'anchor_score_test_summary.csv')
with open(out_file, 'w') as out:
    out.write('idx,anchor,n_samps,n_targs,one_sided_test_stat,two_sided_test_stat,one_sided_p_val,two_sided_p_val\n')
    files = os.listdir(args.json_dir)
    for i in tqdm(range(len(files))):
        f = files[i]
        if f.endswith('.json'):
            f_path = os.path.join(args.json_dir, f)
            ## read json
            file = open(f_path)
            json_str = file.read()
            file.close()
            json_obj = json.loads(json_str)
            ## extract relevant information
            anchor = os.path.splitext(f)[0]
            anchor_idx = anchor.split('_')
            n_samps = int(json_obj['model']['n_samps'])
            n_targs = int(json_obj['model']['n_targs'])
            one_sided_t = float(json_obj['test']['one_side'])
            two_sided_t = float(json_obj['test']['two_side'])
            one_sided_p = float(json_obj['test']['one_side_p'])
            two_sided_p = float(json_obj['test']['two_side_p'])
            ## write to file
            out.write(f'{anchor_idx[0]},{anchor_idx[-1]},{n_samps},{n_targs},{one_sided_t},{two_sided_t},{one_sided_p},{two_sided_p}\n')

## add BY corrected p-values
file_conts = pd.read_csv(out_file)
file_conts['one_sided_p_val_by'] = fdc(
    ps = file_conts['one_sided_p_val'],
    method = 'by'
)
file_conts['two_sided_p_val_by'] = fdc(
    ps = file_conts['two_sided_p_val'], 
    method = 'by'
)
## join with SPLASH output 
splash_output = os.path.join(args.splash_out_dir, 'result.after_correction.all_anchors.tsv')
splash_df = pd.read_csv(splash_output, sep = '\t', low_memory = False)
all_join = file_conts.merge(splash_df, how = 'left', on = 'anchor')
all_join.to_csv(out_file, index = False)


