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

## sites with at least 20 observations in 2020-12-01 to 2021-03-31
sites = {
    'Abwasserverband Wörthersee West|ATTP_2-K1944050': 'worthersee_west',
    'ARA Zeller Becken|ATTP_5-A1844668R161': 'zeller_becken',
    'Bregenz|ATTP_8-Bregenz': 'bregenz',
    'EMREG_BE_Abwasserverband Region Hohenems|ATTP_8-Hohenems': 'hohenems',
    'EMREG_BE_ebswien hauptkläranlage|ATTP_9-HKA-Simmering': 'simmering',
    'EMREG_BE_Hauptsitz AIZ Strass|ATTP_7-7930111': 'aiz_strass',
    'EMREG_BE_Kläranlage Pfarrwerfen|ATTP_5-A1779244R157': 'pfarrwerfen',
    'EMREG_BE_Siggerwiesen_2|ATTP_5-A1699622R154': 'siggerwiesen',
    'Innsbruck|ATTP_7-7101301': 'innsbruck',
    'Kufstein|ATTP_7-7513103': 'kufstein',
    'Stadtgemeinde Graz ARA Graz Gössendorf 1/316|ATTP_6-M3489301R0': 'gossendorf'
}

## convert collection date to integer days after first colleciton date recorded
coll_dates = pd.to_datetime(metadata['collection date'])
metadata['date'] = coll_dates - min(coll_dates)
metadata['date'] = metadata['date'].dt.days

## convert location to text
metadata.rename(columns = {'name of the sampling site': 'site'}, inplace = True)
metadata['site'] = metadata['site'].replace(sites)

## join with SPLASH sample IDs for easy access later
metadata = metadata.merge(sample_ids, left_on = 'srr_id', right_on = 'srr')

## remove biosample ID (duplicated as samn_id)
metadata = metadata.drop(columns = ['biosample_id'])
metadata.to_csv(os.path.join(args.output, 'clean_metadata.tsv'), sep = '\t', index = False, header = True)

