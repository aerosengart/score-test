#!/usr/bin/env python

"""
filter.py - create input_samples.txt. 
ARGS: 
    meta     - str; path to metadata.tsv file
    ids      - str; path to ids.txt file
    output   - str; path to output directory
    min_date - str; first collection date to filter on
    max_date - str; last collection date to filter on
RAISES:  None
RETURNS: None
"""

import argparse
import os
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--meta", type = str, help = "path to metadata.tsv")
parser.add_argument("--ids", type = str, help = "path to ids.txt")
parser.add_argument("--output", type = str, help = "path to output directory")
args = parser.parse_args()

## read in files
metadata = pd.read_csv(args.meta, sep = '\t', header = 0)
ids = pd.read_csv(args.ids, header = None)
ids.columns = ['srr_id', 'sra_id', 'samn_id']
non_miss = metadata.merge(ids, right_on = 'srr_id', left_on = 'Sequence Read Run')
non_miss = non_miss[metadata['Collection Date'] != 'missing'] 
non_miss = non_miss[metadata['Collection Site'] != 'missing']


## write to file
non_miss.to_csv(os.path.join(args.output, 'metadata_filt.tsv'), sep = '\t', index = False, header = True)

## join with IDs and write to file
selected_ids = non_miss['srr_id']
selected_ids.to_csv(os.path.join(args.output, 'input_samples.txt'), index = False, header = False)
