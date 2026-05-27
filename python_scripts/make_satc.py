#!/usr/bin/env python

"""
make_satc.py - create anchor-specific S(A)TC files.

- creates 'anchor_satc' sub-directory in scratch directory
- creates 'all_anchors_list.txt' file in output directory
- dumps binary files as SATC files 
- writes separate anchor-specific S(A)TC files as [ANCHOR SEQUENCE].txt in 'anchor_satc' sub-directory
- writes [ANCHOR SEQUENCE] as line in 'all_anchors_list.txt'
- deletes dumped binary files

ARGS:
    args.sp_dir  - str; directory to SPLASH executables
    args.bin_dir - str; directory of binary files
    args.scratch - str; scratch directory
    args.output  - str; output directory
RAISES: None
RETURNS: None
"""
import argparse
import os
import csv
import subprocess
import glob

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--sp_dir", type = str, help = "path to splash directory")
parser.add_argument("--bin_dir", type = str, help = "path to bin.satc directory")
parser.add_argument("--scratch", type = str, help = "path to scratch directory for temporary file writing")
parser.add_argument("--output", type = str, help = "path to save output directory")
args = parser.parse_args()

## check if save paths exists
output_file = os.path.join(args.output, 'all_anchors_list.txt')
if not os.path.isdir(args.output):
    os.makedirs(args.output)
scratch_dir = os.path.join(args.scratch, 'anchor_satc')
if not os.path.isdir(scratch_dir):
    os.makedirs(scratch_dir)

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

        reader = csv.reader(satc_file, delimiter = '\t')
        for line in reader:
            samp_idx    = line[0]
            anchor      = line[1]
            target      = line[2]
            count       = line[3].rstrip()

            ## write line to anchor-specific SATC file
            with open(os.path.join(scratch_dir, f'{anchor}.txt'), 'a') as f:
                f.write(f'{samp_idx} {target} {count}\n')
            ## write line to all anchors list
            with open(output_file, 'a') as f:
                f.write(f'{anchor}\n')

    ## delete dump file
    subprocess.call(['rm', bin_dump])
## delete dump subdirectory
subprocess.call(['rm', '-r', os.path.join(args.bin_dir, 'satc_dump')])

        