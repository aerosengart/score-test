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

import pandas as pd

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--sp_dir", type = str, help = "path to splash directory")
parser.add_argument("--bin_dir", type = str, help = "path to bin.satc directory")
parser.add_argument("--bin_key", type = str, help = "path to anchor_bin_key.csv")
parser.add_argument("--output", type = str, help = "path to save output directory")
parser.add_argument("--scratch", type = str, help = "path to scratch directory for temporary file writing")
parser.add_argument("--metadata", type = str, help = "path to clean_metadata.tsv")
#parser.add_argument("--covar", type = str, help = "covariate to filter on")
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
covar_filt_file = os.path.join(args.output, f'anchor_filt_summary.csv')
with open(anchor_summ_file, 'w') as out_file:
    out_file.write('bin,anchor,total_count,total_num_nonzero,num_targets,num_samples,tab_prop_nonzero,max_count,satc_file\n')
## get unique values of covariate
md = pd.read_csv(args.metadata, sep = '\t', header = 0, index_col = 0)

## add headers to output files
with open(covar_filt_file, 'w') as out_file:
    out_file.write('bin,anchor,total_count,total_num_nonzero,num_targets,num_samples,num_cand_targs,tab_prop_nonzero,max_count\n')

## loop over all binary files
bin_files = glob.glob(os.path.join(args.bin_dir, '*.satc'))
for i in tqdm(range(len(bin_files))):
    bin_file = bin_files[i]
    ## dump bin to satc
    bin_read = os.path.join(args.bin_dir, f'bin{i}.satc')
    bin_dump = os.path.join(args.output, 'satc_dump', f'bin{i}.satc.dump')
    if not os.path.isfile(bin_dump):
        subprocess.call([args.sp_dir + '/satc_dump', bin_read, bin_dump])
    delete_files = []
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

            ## write line to anchor-specific SATC file
            anchor_file = os.path.join(scratch_dir, f'{anchor}.txt')
            with open(anchor_file, 'a') as f:
                f.write(f'{samp_idx} {target} {count}\n')

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
                ## list of maximum count per row
                anchor_targ_max_count = []
                ## list of all samples from all targets in anchor table
                anchor_samples = []
                for targ in anchor_dict.keys():
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
                    ######## anchor table summary statistics
                    ## add target row counts to anchor table counts
                    anchor_counts  += targ_counts
                    ## add number of non-zero cells in target row to anchor table non-zero cells
                    anchor_nonzero += num_nonzero
                    ## add non-zero sample IDs to anchor table non-zero sample IDs list
                    anchor_samples += targ_samples
                    ## add number of non-zero cells in target row to anchor table list
                    anchor_targ_nonzero += [len(counts)]
                    ## add maximum count for target row
                    anchor_targ_max_count += [max(counts)]

                ## get unique sample IDs
                unique_samples = set(anchor_samples)
                ## get proportion of non-zero cells for entire anchor table
                num_targs = len(anchor_dict.keys())
                tab_prop_nonzero = anchor_nonzero / (len(unique_samples) * num_targs)
                ## get maximum count in table
                max_count = max(anchor_targ_max_count)
                ## write to filtered summary file ##
                md_samps = md[md['splash_id'].astype(str).isin(unique_samples)]
                ## number of observations per site
                sd_counts = md_samps.groupby(['site']).size()
                sd_counts = sd_counts[sd_counts >= 25].index.tolist()
                ## number of observations per lab
                lab_counts = md_samps.groupby(['sample_source_institute']).size()
                lab_counts = lab_counts[lab_counts >= 25].index.tolist()
                ## compute candidate targets
                cand_targs = [x for x in anchor_targ_max_count if x > 10]
                ## only write to file if ALL conditions pass
                if len(md_samps) > 300 \
                    and len(cand_targs) >= 50 \
                        and len(sd_counts) > 1 \
                            and len(lab_counts) > 1:
                    with open(covar_filt_file, 'a') as out_file:
                        out_file.write(f'{i},{anchor},{anchor_counts},{anchor_nonzero},{num_targs},{len(unique_samples)},{len(cand_targs)},{tab_prop_nonzero},{max_count}\n')
                    kept_satc = 'saved'
                else:
                    delete_files += [anchor_file]
                    kept_satc = 'deleted'
                ## write to all anchors file and whether SATC was saved
                with open(anchor_summ_file, 'a') as out_file:
                    out_file.write(f'{i},{anchor},{anchor_counts},{anchor_nonzero},{num_targs},{len(unique_samples)},{tab_prop_nonzero},{max_count},{kept_satc}\n')
                ## update line
                last_anchor = anchor
                anchor_dict = {target: {samp_idx: count}}

    ## delete files to save space
    for f in delete_files:
        print(f'Deleting {len(delete_files)} SATC files...\n')
        subprocess.call(['rm', f])

    ## delete bin_dump file to save space 
    if args.clean_up:
        subprocess.call(['rm', bin_dump])
    