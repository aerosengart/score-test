#!/usr/bin/env python

"""
satc_filt.py - Filter anchor satc files according to anchor_filt.csv. 

ARGS:
    args.satc_dir    - str; path to anchor-specific satc files
    args.anchor_file - str; path to anchor_filt_summary.csv file
    args.output      - str; args.output directory
    args.clean_up    - bool; whether to remove satc_dir after run
RAISES: None
RETURNS: None
"""
import argparse
import csv
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--satc_dir", type = str, help = "path to anchor-specific satc files")
parser.add_argument("--anchor_file", type = str, help = "path to anchor_filt_summary.csv")
parser.add_argument("--output", type = str, help = "path to save output directory")
parser.add_argument("--clean_up", action = "store_true", help = "delete satc_dir")
args = parser.parse_args()

## check if save path exists
anch_satc_dir = os.path.join(args.output, 'anchor_satc')
if not os.path.isdir(anch_satc_dir):
    os.makedirs(anch_satc_dir)
## list of anchors for score test
anch_list_file = os.path.join(args.output, 'anchor_list.txt')

## move anchors in filtered list to home directory
with open(args.anchor_file, 'r') as anch_file:
    header = next(anch_file)
    reader = csv.reader(anch_file)
    for line in reader:
        bin         = line[0]
        anchor      = line[1]
        anch_satc_file = os.path.join(anch_satc_dir, f'{anchor}.txt')
        subprocess.call(
            ['cp', 
             os.path.join(args.satc_dir, f'{anchor}.txt'), 
             anch_satc_file]
        )
        with open(anch_list_file, 'a') as list_file:
            list_file.write(f'{anchor}\n')

## delete the satc_dir (so many files and we no longer need them)
if args.clean_up:
    subprocess.call(['rm', '-r', args.satc_dir])

