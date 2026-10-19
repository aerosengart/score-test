#!/usr/bin/env python

"""
split_bin_key.py - split anchor_bin_key.csv into separate files containing the specified bin indices
"""
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--key", type = str, help = "path to anchor_bin_key.csv")
parser.add_argument("--start_bin", type = int, help = "first bin index")
parser.add_argument("--end_bin", type = int, help = "last bin index")
parser.add_argument("--output", type = str, help = "path to output directory")
args = parser.parse_args()

## checks
assert args.start_bin <= args.end_bin
assert args.start_bin >= 0

## make output directory
out_dir = os.path.join(args.output, 'split_bins')
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)
out_file = os.path.join(out_dir, f'anchor_bin_key_{args.start_bin}_{args.end_bin}.csv')

## make output file
with open(out_file, 'w') as f:
    f.write('anchor,bin,start_idx,end_idx\n')

with open(args.key, 'r') as fp:
    ## skip header
    next(fp)
    for i, line in enumerate(fp):
        spl_line = line.strip().split(',')
        if int(spl_line[1]) < args.start_bin:
            continue
        elif int(spl_line[1]) > args.end_bin:
            break
        else:
            with open(out_file, 'a') as f:
                f.write(line)



