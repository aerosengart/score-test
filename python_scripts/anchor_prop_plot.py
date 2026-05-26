#!/usr/bin/env python

"""
anchor_prop.py - Create list of all anchors, the proportion of zero cells in their contingency tables, and the total counts in their tables.

ARGS:
    args.sp_dir  - str; directory to SPLASH executables
    args.bin_dir - str; directory of binary files
    args.bin_key - str; path to anchor_bin_key.csv
    args.output  - str; args.output directory
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
parser.add_argument("--anchor_file", type = str, help = "path to anchor_filt_summary.csv")
args = parser.parse_args()

## check if save path exists
out_path = os.path.join(args.output, 'plots')
if not os.path.isdir(out_path):
    os.makedirs(out_path)

## read in anchor file
anch = pd.read_csv(args.anchor_file, header = 0)

## anchor_filt_summary.csv header has 7 columns to plot
header_vals = ['total_count', 'total_num_nonzero', 'num_targets', \
               'num_samples', 'tab_prop_nonzero', 'med_row_prop_nonzero']
plot_titles = ['Total Cell Count per Table', 
               'Non-Zero Cell Count per Table',
               'Number of Targets per Table',
               'Number of Samples per Table',
               'Proportion of Non-Zero Cells per Table',
               'Median Target (Row) Proportion of Non-Zero Cells per Table']
axis_labels = ['count', 'count', 'count', 'count', \
               'proportion', 'proportion']

plt.rcParams.update({'font.size': 8})
log10_vec = np.vectorize(math.log10)

for i in range(len(header_vals)):
    val   = header_vals[i]
    lab   = axis_labels[i]
    title = plot_titles[i]
    plt_name = os.path.join(out_path, f'{val}_hist.pdf')
    col_data = anch[val].tolist()
    if max(col_data) > 1e4:
        x_vals = log10_vec(col_data)
        x_lab  = f'{lab} ($\\log_{{10}}$ scale)'
    else:
        x_vals = col_data
        x_lab  = lab

    fig, axs = plt.subplots(tight_layout = True)
    axs.set_title(title)
    axs.set_ylabel('density')
    axs.set_xlabel(x_lab)
    axs.hist(x_vals, bins = 100, density = True)
    plt.savefig(plt_name)
    plt.close()


