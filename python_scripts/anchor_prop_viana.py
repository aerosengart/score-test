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

import pandas as pd

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--sp_dir", type = str, help = "path to splash directory")
parser.add_argument("--bin_dir", type = str, help = "path to bin.satc directory")
parser.add_argument("--bin_key", type = str, help = "path to anchor_bin_key.csv")
parser.add_argument("--output", type = str, help = "path to save output directory")
parser.add_argument("--scratch", type = str, help = "path to scratch directory for temporary file writing")
parser.add_argument("--metadata", type = str, help = "path to clean_metadata.tsv")
parser.add_argument("--covar", type = str, default = None, help = "covariate to filter on")
parser.add_argument("--clean_up", action = "store_true", help = "remove dump files")
args = parser.parse_args()

## check if save path exists
if not os.path.isdir(args.output):
    os.makedirs(args.output)
if not os.path.isdir(os.path.join(args.output, 'satc_dump')):
    os.makedirs(os.path.join(args.output, 'satc_dump'))
scratch_dir = os.path.join(args.scratch, 'anchor_satc')
if not os.path.isdir(scratch_dir):
    os.makedirs(scratch_dir)

## output file
anchor_bin_file  = os.path.join(args.bin_key)
anchor_summ_file = os.path.join(args.output, 'anchor_summary.csv')
anchor_filt_file = os.path.join(args.output, 'anchor_filt_summary.csv')

## check if filtering on covariate as well
if args.covar is not None:
    covar_filt_file = os.path.join(args.output, f'anchor_filt_by_{args.covar}.csv')
    with open(covar_filt_file, 'w') as out_file:
        out_file.write(
            'bin,anchor,num_targets,num_samples,total_count,valid_targs\n'
        )
    ## get unique values of covariate
    md = pd.read_csv(args.metadata, sep = '\t', header = 0, index_col = 0)
    unique_covar = md[args.covar].unique()

## add headers to output files
with open(anchor_summ_file, 'w') as out_file:
    out_file.write(
        'bin,anchor,num_targets,num_samples,total_count,valid_targs\n'
    )
with open(anchor_filt_file, 'w') as out_file:
    out_file.write(
        'bin,anchor,num_targets,num_samples,total_count,valid_targs\n'
    )

## loop over all binary files
bin_files = glob.glob(os.path.join(args.bin_dir, '*.satc'))
for i in tqdm(range(len(bin_files))):
    bin_file = bin_files[i]
    ## dump bin to satc
    bin_read = os.path.join(args.bin_dir, f'bin{i}.satc')
    bin_dump = os.path.join(args.output, 'satc_dump', f'bin{i}.satc.dump')
    if not os.path.isfile(bin_dump):
        subprocess.call([args.sp_dir + '/satc_dump', bin_read, bin_dump])
    ## read bin.satc.dump line by line
    with open(bin_dump, 'r') as satc_file:
        first_row = next(satc_file).split('\t')
        samp_idx = first_row[0]
        anchor = first_row[1]
        target = first_row[2]
        count = first_row[3].rstrip()

        last_anchor = anchor
        anchor_dict = {target: {samp_idx: count}}
        
        reader = csv.reader(satc_file, delimiter = '\t')
        for line in reader:
            samp_idx    = line[0]
            anchor      = line[1]
            target      = line[2]
            count       = line[3].rstrip()

            if anchor == last_anchor:
                ## same anchor as before - add data to dictionary
                if target in anchor_dict.keys():
                    ## append if target already in sub-dictionary
                    anchor_dict[target][samp_idx] = count
                else:
                    ## add new sub-dictionary if not
                    anchor_dict.update({target: {samp_idx: count}})
            else:
                ## new anchor — write anchor data to file and reset dictionary
                ## total counts for anchor table
                anchor_counts = 0
                ## number of non-zero cells in anchor table
                anchor_nonzero = 0
                ## list of non-zero cells by row (len == # targets in anchor table)
                anchor_targ_nonzero = []
                ## list of all samples from all targets in anchor table
                anchor_samples = []
                ## list of target indices that are valid
                valid_targs = []
                for idx, targ in enumerate(anchor_dict.keys()):
                    ######## target row summary statistics
                    ## sample IDs for target row in anchor table
                    targ_samples = [samp for samp in anchor_dict[targ].keys()]
                    ## target-sample counts
                    counts = [int(value) for value in anchor_dict[targ].values()]
                    ## total number of sample with non-zero values in target row
                    num_samps    = len(targ_samples)
                    ## total count for target row
                    targ_counts  = sum(counts)
                    ## number of non-zero cells in target row
                    num_nonzero  = len(counts)
                    ## add target row counts to anchor table counts
                    anchor_counts  += targ_counts
                    ## add number of non-zero cells in target row to anchor table non-zero cells
                    anchor_nonzero += num_nonzero
                    ## add non-zero sample IDs to anchor table non-zero sample IDs list
                    anchor_samples += targ_samples
                    ## add number of non-zero cells in target row to anchor table list
                    anchor_targ_nonzero += [len(counts)]
                    ## valid target if maximum count in row is greater than 5
                    if max(counts) > 5:
                        valid_targs += [idx]
                        ## write SAMPLE TARGET COUNT to anchor-specific SATC file if target is valid
                        with open(os.path.join(scratch_dir, f'{anchor}.txt'), 'a') as f:
                            for i in range(len(targ_samples)):
                                f.write(f'{targ_samples[i]} {targ} {counts[i]}\n')

                ## get unique sample IDs
                unique_samples = set(anchor_samples)
                ## total number of targets in anchor table
                num_targs = len(anchor_dict.keys())

                ## write to overall summary file
                with open(anchor_summ_file, 'a') as out_file:
                    out_file.write(f'{i},{anchor},{num_targs},{len(unique_samples)},{anchor_counts},{valid_targs}\n')
                ## write anchor stats to filtered .csv if there are at least 5 valid rows and 20 samples
                if len(unique_samples) >= 20 and len(valid_targs) >= 5:
                    with open(anchor_filt_file, 'a') as out_file:
                        out_file.write(f'{i},{anchor},{num_targs},{len(unique_samples)},{anchor_counts},{valid_targs}\n')
                    ## if filtering by covariate, do necessary checks/writing
                    if args.covar is not None:
                        md_samps = md[md['splash_id'].astype(str).isin(unique_samples)]
                        md_samps_unique_covar = md_samps[args.covar].unique()
                        ## only write if the covariate of interest has at least 2 different values
                        if len(md_samps_unique_covar) >= 2:
                            with open(covar_filt_file, 'a') as out_file:
                                out_file.write(f'{i},{anchor},{num_targs},{len(unique_samples)},{anchor_counts},{valid_targs}\n')
                ## reset dictionary
                last_anchor = anchor
                anchor_dict = {target: {samp_idx: count}}

    ## delete bin_dump file to save space 
    if args.clean_up:
        subprocess.call(['rm', bin_dump])
    