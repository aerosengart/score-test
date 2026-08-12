#!/usr/bin/env python

"""
metadata.py - create metadata.tsv for BioProject
REQUIRES: sh_dir - path to shell script directory
          bsl    - path to BioSample ID list
          head   - path to .tsv file of header names
          output - output directory
MODIFIES: None
RETURNS: None
"""

import argparse
import itertools
import os
import numpy as np
import pandas as pd
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--sh_dir", type = str, help = "path to shell script directory")
parser.add_argument("--bsl", type = str, help = "path to BioSample ID list")
parser.add_argument("--head", type = str, help = "path to .tsv file of header names")
parser.add_argument("--output", type = str, help = "path to save output")
parser.add_argument("--european", action = "store_true", help = "European experiment?")
args = parser.parse_args()

## create temporary metadata_temp.csv if it doesn't exists already
temp_md = os.path.join(args.output, "metadata_temp.tsv")
if not os.path.isfile(temp_md):
  if args.european:
    cmd = "./" + os.path.join(args.sh_dir, "metadata_e.sh")
  else:
    cmd = "./" + os.path.join(args.sh_dir, "metadata.sh")
  subprocess.run([cmd, "-l", args.bsl, "-o", args.output])
else:
  print("Metadata already fetched!")

headers = pd.read_csv(args.head, sep = '\t', header = None).values.flatten().tolist()
headers = [x.strip() for x in headers]
new_data = []

## iterate by 3 rows to get column names and data at the same time (third line is empty)
with open(temp_md) as f:
  for cols, data, empty in itertools.zip_longest(*[f]*3):
    cols = cols.split('\t')
    cols = [x.strip(' ').strip('\n') for x in cols] ## strip whitespace

    data = data.split('\t')
    data = [x.strip(' ').strip('\n') for x in data] ## strip whitespace

    ## iterate through data available for this sample
    new_row = []
    for element in headers:
      try:
        index = cols.index(element)
        index_value = data[index]
      except ValueError:
        index_value = 'NA'
      new_row.append(index_value)
    new_data.append(new_row)
  

md = pd.DataFrame(new_data, columns = headers)
md.to_csv(os.path.join(args.output, "metadata.tsv"), sep = "\t")
