#!/usr/bin/env python

"""
clean_metadata.py - clean metadata .tsv for easier ingestion into Data/Model classes

ARGS: 
    meta     - str; path to metadata_filt.tsv file used in SPLASH
    id_key   - str; path to sample_name_to_id.mapping.txt from SPLASH
    output   - str; path to output directory
RAISES:  None
RETURNS: None
"""

import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--meta",   type = str, help = "path to metadata_filt.tsv")
parser.add_argument("--id_key", type = str, help = "path to sample_name_to_id.mapping.txt")
parser.add_argument("--output", type = str, help = "path to output directory")
args = parser.parse_args()

## read in files
metadata   = pd.read_csv(args.meta, sep = '\t', header = 0, index_col = 0)
sample_ids = pd.read_csv(args.id_key, sep = ' ', header = None, names = ['srr', 'splash_id'])

## convert collection date to integer days after first colleciton date recorded
coll_dates = pd.to_datetime(metadata['collection_date'])
metadata['date'] = coll_dates - min(coll_dates)
metadata['date'] = metadata['date'].dt.days

## refactor infections
metadata = metadata.replace({'Omicron(BA.1)': 'ba1', 'Omicron(BA.2)': 'ba2', 'Delta': 'delta'})

## join with SPLASH sample IDs for easy access later
metadata = metadata.merge(sample_ids, left_on = 'srr_id', right_on = 'srr')

## remove biosample ID (duplicated as samn_id)
metadata = metadata.drop(columns = ['biosample_id'])
metadata.to_csv(os.path.join(args.output, 'clean_metadata.tsv'), sep = '\t', index = False, header = True)

