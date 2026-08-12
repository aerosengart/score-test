#!/usr/bin/env python

"""
list_anchors.py - Create list of anchors from a provided filtered anchor summary .csv or .tsv file.

ARGS:
    args.output      - str; args.output directory
    args.anchor_file - str; path to filtered anchor file
RAISES: None
RETURNS: None
"""
import argparse
import csv
import os
import subprocess
import glob
import statistics
import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--output", type = str, help = "path to save output directory")
parser.add_argument("--anchor_file", type = str, help = "path to filtered anchor file")
parser.add_argument("--satc_dir", type = str, help = "path to anchor satc files", required = False)
args = parser.parse_args()

## check if save path exists
out_path = os.path.join(args.output)
if not os.path.isdir(out_path):
    os.makedirs(out_path)

## read in anchor file
anch = pd.read_csv(args.anchor_file, header = 0)
## copy desired anchor satc files from scratch if satc_dir is passed as argument
if args.satc_dir is not None:
    if not os.path.isdir(os.path.join(args.output, 'anchor_satc')):
        os.makedirs(os.path.join(args.output, 'anchor_satc'))
    for anchor in anch.anchor:
        read_path = os.path.join(args.satc_dir, f'{anchor}.txt')
        save_path = os.path.join(args.output, 'anchor_satc', f'{anchor}.txt')
        subprocess.call(['cp', read_path, save_path])
anch.anchor.to_csv(os.path.join(out_path, 'anchor_list.txt'), index = False, header = False)
