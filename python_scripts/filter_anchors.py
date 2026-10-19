#!/usr/bin/env python

import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--splash_file", type = str, help = "path to result.after_correction.all_anchors.tsv")
parser.add_argument("--anchor_bin_file", type = str, help = "path to anchor_bin_key.csv")
parser.add_argument("--m_thresh", type = int, help = "total anchor table count minimum")
parser.add_argument("--uniq_targs", type = int, help = "number of unique targets minimum")
parser.add_argument("--nonzero_samps", type = int, help = "number of non-zero samples minimum")
parser.add_argument("--output", type = str, help = "path to output directory")
args = parser.parse_args()


## read in files
result_df = pd.read_csv(args.splash_file, sep = '\t', header = 0)
anchor_bin_file = pd.read_csv(args.anchor_bin_file, sep = ',', header = 0)

## subset
result_df = result_df[result_df['anch_uniqTargs'] >= args.uniq_targs]
result_df = result_df[result_df['number_nonzero_samples'] >= args.nonzero_samps]
result_df = result_df[result_df['M'] >= args.m_thresh]

## filter to those anchors
anchors = result_df['anchor']
anchor_bin_file = anchor_bin_file[anchor_bin_file['anchor'].isin(anchors)]

## check if save path exists
if not os.path.isdir(args.output):
    os.makedirs(args.output)
anchor_bin_file.to_csv(os.path.join(args.output, 'anchor_bin_key_filter.csv'), index = False)

