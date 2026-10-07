#!/usr/bin/env python

"""
anchor_bin_key.py - Create list of all anchors and the corresponding binarized satc file index. 

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

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--sp_dir", type = str, help = "path to splash directory")
parser.add_argument("--bin_dir", type = str, help = "path to bin.satc directory")
parser.add_argument("--output", type = str, help = "path to save output directory")
parser.add_argument("--clean_up", action = "store_true", help = "remove dump files")
args = parser.parse_args()

## check if save path exists
if not os.path.isdir(os.path.join(args.output, 'satc_dump')):
    os.makedirs(os.path.join(args.output, 'satc_dump'))

## output file
anchor_bin_file = os.path.join(args.output, 'anchor_bin_key.csv')
with open(anchor_bin_file, "w") as f:
    f.write(f'anchor,bin,start_idx,end_idx\n')

## loop over all binary files
bin_files = glob.glob(os.path.join(args.bin_dir, '*.satc'))
with open(anchor_bin_file, "a") as f:
    ## loop over binary files
    for i in range(len(bin_files)):
        ## dump satc files from bin
        bin_read = os.path.join(args.bin_dir, f'bin{i}.satc')
        bin_dump = os.path.join(args.output, 'satc_dump', f'bin{i}.satc.dump')
        if not os.path.exists(bin_dump):
            subprocess.call([args.sp_dir + '/satc_dump', bin_read, bin_dump])
        with open(bin_dump, 'r') as file:
            first_line = file.readline().strip('\n').split('\t')
            curr_idx = 1
            curr_anch = first_line[1]
            start_idx = 0
            end_idx = 0
            for line in file:
                spl_line = line.strip('\n').split('\t')
                anch = spl_line[1]
                if anch != curr_anch:
                    ## don't need to subtract 1 from curr_idx because it is already 1 behind the "for line" loop due to the first "readline()" call
                    new_line = f'{curr_anch},{i},{start_idx},{curr_idx - 1}\n'
                    start_idx = curr_idx
                    curr_anch = anch
                    with open(anchor_bin_file, 'a') as f:
                        f.write(new_line)
                curr_idx += 1
            subprocess.call(['rm', bin_dump])
            
        
        # satc = pd.read_csv(bin_dump, sep = '\t')
        # ## rename columns
        # if satc.shape[-1] == 4:
        #     satc.columns = ['sample', 'anchor', 'target', 'count']
        # else: ## 10X data
        #     satc.columns = ['sample_id', 'cell_barcode', 'anchor', 'target', 'count']
        # anchors = satc['anchor'].unique()
        # for i in tqdm(range(len(anchors))):
        #     anchor = anchors[i]
        #     first_idx = satc['anchor'].eq(anchor).idxmax()
        #     last_idx = satc['anchor'].eq(anchor).idxmin()
        #     with open(anchor_bin_file, "a") as f:
        #         f.write(f'{anchor},{i},{first_idx},{last_idx}\n')
        ## delete bin_dump file to save space 

if args.clean_up:
    subprocess.call(['rm', '-r', os.path.join(args.bin_dir, 'satc_dump')])

