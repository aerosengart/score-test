#!/usr/bin/env python

"""
filter.py - create input_samples.txt. 
ARGS: 
    meta     - str; path to metadata.tsv file
    ids      - str; path to ids.txt file
    output   - str; path to output directory
    min_date - str; first collection date to filter on
    max_date - str; last collection date to filter on
    labs     - str; path to processing laboratory labels file
RAISES:  None
RETURNS: None
"""

import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--meta", type = str, help = "path to metadata.tsv")
parser.add_argument("--ids", type = str, help = "path to ids.txt")
parser.add_argument("--output", type = str, help = "path to output directory")
parser.add_argument("--min_date", type = str, help = "first collection date")
parser.add_argument("--max_date", type = str, help = "last collection date")
parser.add_argument("--labs", type = str, help = "path to labs file")
args = parser.parse_args()

# ## sites with at least 20 observations in 2020-12-01 to 2021-03-31
# sites = [
#     'Abwasserverband Wörthersee West|ATTP_2-K1944050',
#     'ARA Zeller Becken|ATTP_5-A1844668R161',
#     'Bregenz|ATTP_8-Bregenz',
#     'EMREG_BE_Abwasserverband Region Hohenems|ATTP_8-Hohenems',
#     'EMREG_BE_ebswien hauptkläranlage|ATTP_9-HKA-Simmering',
#     'EMREG_BE_Hauptsitz AIZ Strass|ATTP_7-7930111',
#     'EMREG_BE_Kläranlage Pfarrwerfen|ATTP_5-A1779244R157',
#     'EMREG_BE_Siggerwiesen_2|ATTP_5-A1699622R154',
#     'Innsbruck|ATTP_7-7101301',
#     'Kufstein|ATTP_7-7513103',
#     'Stadtgemeinde Graz ARA Graz Gössendorf 1/316|ATTP_6-M3489301R0'
# ]

## read in files
metadata = pd.read_csv(args.meta, sep = '\t', header = 0)
ids = pd.read_csv(args.ids)
labs = pd.read_csv(args.labs, sep = '\t', header = 0)
ids.columns = ['srr_id', 'sra_id', 'samn_id'] ## technically these have different names for the european data
## filter to non-missing locations and dates in the desired timeframe and sites
non_miss = metadata[metadata['name of the sampling site'] != 'missing']
# non_miss = metadata[metadata['name of the sampling site'].isin(sites)]
non_miss = non_miss[metadata['collection date'] != 'missing'] ## line added 2026-05-19
non_miss['collection date'] = pd.to_datetime(non_miss['collection date'], format = '%Y-%m-%d')
non_miss = non_miss[non_miss['collection date'] >= args.min_date]
non_miss = non_miss[non_miss['collection date'] <= args.max_date]

## join with IDs
non_miss = non_miss.merge(ids, left_on = 'sra_id', right_on = 'sra_id')

## join with labs and write to file
non_miss = non_miss.merge(labs, left_on = 'srr_id', right_on = 'run_accession')
non_miss.to_csv(os.path.join(args.output, 'metadata_filt.tsv'), sep = '\t', index = False, header = True)

selected_ids = non_miss['srr_id']
selected_ids.to_csv(os.path.join(args.output, 'input_samples.txt'), index = False, header = False)
