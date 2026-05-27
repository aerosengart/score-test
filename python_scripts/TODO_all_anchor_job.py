#!/usr/bin/env python

"""
anchor_bin_key.py - Create list of all anchors and the corresponding binarized .satc file index. 

ARGS:
    args.sp_dir - str; directory to SPLASH executables
    args.bin_dir    - str; directory of binary files
    args.output    - str; args.output directory
RAISES: None
RETURNS: None
"""
import argparse
import os
import pandas as pd
import subprocess
import glob
import linecache

parser = argparse.ArgumentParser()
parser.add_argument("--satc_dir", type = str, help = "path to anchor-specific S(A)TC directory")
parser.add_argument("--output", type = str, help = "path to save output directory")
parser.add_argument("--anchor_list", type = str, help = "path to anchor list")
parser.add_argument("--start", type = int, help = "starting index for anchor selection (inclusive)")
parser.add_argument("--end", type = int, help = "ending index for anchor selection (not inclusive)")
#parser.add_argument("--clean_up", action = "store_true", help = "remove dump files")
args = parser.parse_args()

## check if save path exists
if not os.path.isdir(os.path.join(args.output, 'anchor_satc')):
    os.makedirs(os.path.join(args.output, 'anchor_satc'))

## get the desired anchor S(A)TC files
for j in range(args.start, args.end):
    ## get anchor

## output file
anchor_bin_file = os.path.join(args.output, 'anchor_bin_key.csv')
with open(anchor_bin_file, "w") as f:
    f.write(f'anchor,bin\n')