#!/usr/bin/env python

"""
align_kmers.py - Align k-mers to reference indexes and determine "strain-defining" anchors/targets

ARGS:
    args.splash_calls - path to splash calls file (splash_calls_filt.csv)
    args.score_calls  - path to score test calls file (score_calls_filt.csv)
    args.output       - output directory
    args.bowtie_dir   - path to Bowtie executables
    args.index        - basename of reference genome index for Bowtie
RAISES: None
RETURNS: None
"""
import argparse
import os
import numpy as np
import pandas as pd
import subprocess
import csv

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--splash_calls", type = str, help = "path to splash calls file")
parser.add_argument("--score_calls", type = str, help = "path to score test calls file")
parser.add_argument("--output", type = str, help = "path to save output directory")
parser.add_argument("--bowtie_dir", type = str, help = "path to bowtie binaries")
parser.add_argument("--index", type = str, help = "path the genome index")
args = parser.parse_args()

if not os.path.isdir(args.output):
    os.makedirs(args.output)

## create comma-separated lists of called anchors/targets
splash_anchs = os.path.join(args.output, 'splash_calls_anchs.csv')
splash_targs = os.path.join(args.output, 'splash_calls_targs.csv')
score_anchs = os.path.join(args.output, 'score_calls_anchs.csv')
score_targs = os.path.join(args.output, 'score_calls_targs.csv')

splash_calls = pd.read_csv(args.splash_calls, header = 0)
uniq_anchors_spl = splash_calls['anchor'].unique().tolist()
uniq_targets_spl = splash_calls['target'].unique().tolist()
with open(splash_anchs, 'w') as of:
    of.write('\n'.join(uniq_anchors_spl))
with open(splash_targs, 'w') as of:
    of.write('\n'.join(uniq_targets_spl))

score_calls = pd.read_csv(args.score_calls, header = 0)
uniq_anchors_sc = score_calls['anchor'].unique().tolist()
uniq_targets_sc = score_calls['target'].unique().tolist()
with open(score_anchs, 'w') as of:
    of.write('\n'.join(uniq_anchors_sc))
with open(score_targs, 'w') as of:
    of.write('\n'.join(uniq_targets_sc))

if (not os.path.isfile(f'{args.output}/splash_anchs.txt')) or \
   (not os.path.isfile(f'{args.output}/splash_targs.txt')) or \
   (not os.path.isfile(f'{args.output}/score_anchs.txt')) or \
   (not os.path.isfile(f'{args.output}/score_targs.txt')):
    print('Aligning all k-mers...')
    ## align k-mers
    subprocess.call([f'{args.bowtie_dir}/bowtie', '-r', '-a', '-v 0', '-x', f'{args.index}', splash_anchs, f'{args.output}/splash_anchs.txt'])
    subprocess.call([f'{args.bowtie_dir}/bowtie', '-r', '-a', '-v 0', '-x', f'{args.index}', splash_targs, f'{args.output}/splash_targs.txt'])
    subprocess.call([f'{args.bowtie_dir}/bowtie', '-r', '-a', '-v 0', '-x', f'{args.index}', score_anchs, f'{args.output}/score_anchs.txt'])
    subprocess.call([f'{args.bowtie_dir}/bowtie', '-r', '-a', '-v 0', '-x', f'{args.index}', score_targs, f'{args.output}/score_targs.txt'])

## filter according to Chaung et al.
align_anchs_spl = pd.read_csv(f'{args.output}/splash_anchs.txt', sep='\t', header=None)
align_anchs_spl.columns = ['idx', 'drctn', 'ref_seq', 'pos', 'anchor', 'qual', 'inst', 'mismatch']
align_targs_spl = pd.read_csv(f'{args.output}/splash_targs.txt', sep='\t', header=None)
align_targs_spl.columns = ['idx', 'drctn', 'ref_seq', 'pos', 'target', 'qual', 'inst', 'mismatch']

align_anchs_sc = pd.read_csv(f'{args.output}/score_anchs.txt', sep='\t', header=None)
align_anchs_sc.columns = ['idx', 'drctn', 'ref_seq', 'pos', 'anchor', 'qual', 'inst', 'mismatch']
align_targs_sc = pd.read_csv(f'{args.output}/score_targs.txt', sep='\t', header=None)
align_targs_sc.columns = ['idx', 'drctn', 'ref_seq', 'pos', 'target', 'qual', 'inst', 'mismatch']

## add columns for alignment to filtered calls dataframes
ref_seqs = align_anchs_spl['ref_seq'].unique().tolist()
for i in range(len(ref_seqs)):
    splash_calls[f'anch_align_{ref_seqs[i]}'] = 0
ref_seqs = align_targs_spl['ref_seq'].unique().tolist()
for i in range(len(ref_seqs)):
    splash_calls[f'targ_align_{ref_seqs[i]}'] = 0
ref_seqs = align_anchs_sc['ref_seq'].unique().tolist()
for i in range(len(ref_seqs)):
    score_calls[f'anch_align_{ref_seqs[i]}'] = 0
ref_seqs = align_targs_sc['ref_seq'].unique().tolist()
for i in range(len(ref_seqs)):
    score_calls[f'targ_align_{ref_seqs[i]}'] = 0

## loop over alignments in each of the four files and fill in call dataframe
if not os.path.isfile(f'{args.output}/splash_calls_alignment.tsv'):
    print('Computing anchor/target alignments for SPLASH...')
    for i in tqdm(range(align_anchs_spl.shape[0])):
        row = align_anchs_spl.iloc[i, :]
        anchor = row['anchor']
        ref_seq = row['ref_seq']
        splash_calls.loc[splash_calls['anchor'] == anchor, f'anch_align_{ref_seq}'] = 1
    for i in tqdm(range(align_targs_spl.shape[0])):
        row = align_targs_spl.iloc[i, :]
        target = row['target']
        ref_seq = row['ref_seq']
        splash_calls.loc[splash_calls['target'] == target, f'targ_align_{ref_seq}'] = 1
    splash_calls.to_csv(f'{args.output}/splash_calls_alignment.tsv', sep = '\t', index = False)
else:
    splash_calls = pd.read_csv(f'{args.output}/splash_calls_alignment.tsv', sep = '\t', header = 0)

if not os.path.isfile(f'{args.output}/score_calls_alignment.tsv'):
    print('Computing anchor/target alignments for score test...')
    for i in tqdm(range(align_anchs_sc.shape[0])):
        row = align_anchs_sc.iloc[i, :]
        anchor = row['anchor']
        ref_seq = row['ref_seq']
        score_calls.loc[score_calls['anchor'] == anchor, f'anch_align_{ref_seq}'] = 1
    for i in tqdm(range(align_targs_sc.shape[0])):
        row = align_targs_sc.iloc[i, :]
        target = row['target']
        ref_seq = row['ref_seq']
        score_calls.loc[score_calls['target'] == target, f'targ_align_{ref_seq}'] = 1
    score_calls.to_csv(f'{args.output}/score_calls_alignment.tsv', sep = '\t', index = False)
else:
    score_calls = pd.read_csv(f'{args.output}/score_calls_alignment.tsv', sep = '\t', header = 0)

## filter according to Chaung et al.
## need anchor/target pairs that align to at least one assembly 
## i.e. anch_* columns sum to at least 1 and targ_* columns sum to at least 1
anch_col_idxs = splash_calls.columns.str.startswith('anch_align')
targ_col_idxs = splash_calls.columns.str.startswith('targ_align')
splash_calls_filt = splash_calls.iloc[splash_calls.iloc[:, anch_col_idxs].sum(axis = 1) > 0, :]
splash_calls_filt = splash_calls_filt.iloc[splash_calls_filt.iloc[:, targ_col_idxs].sum(axis = 1) > 0, :]

anch_col_idxs = score_calls.columns.str.startswith('anch_align')
targ_col_idxs = score_calls.columns.str.startswith('targ_align')
score_calls_filt = score_calls.iloc[score_calls.iloc[:, anch_col_idxs].sum(axis = 1) > 0, :]
score_calls_filt = score_calls_filt.iloc[score_calls_filt.iloc[:, targ_col_idxs].sum(axis = 1) > 0, :]


strain_defn_spl = os.path.join(f'{args.output}/splash_strain_def.tsv')
with open(strain_defn_spl, 'w') as of:
    of.write(f"{'\t'.join(splash_calls.columns)}\n")

strain_defn_sc = os.path.join(f'{args.output}/score_strain_def.tsv')
with open(strain_defn_sc, 'w') as of:
    of.write(f"{'\t'.join(score_calls.columns)}\n")

## "strain-defining" anchors are those that have at least one target mapping to a reference assembly and at least one target that does not 
print('Determining strain-defining anchors for SPLASH...')
uniq_splash_anchs = splash_calls_filt['anchor'].unique()
targ_col_idxs     = splash_calls.columns.str.startswith('targ_align')

for i in tqdm(range(len(uniq_splash_anchs))):
    anchor = uniq_splash_anchs[i]
    anch_calls_filt = splash_calls_filt[splash_calls_filt['anchor'] == anchor]
    ## number of targets associated with anchor
    n_targs = anch_calls_filt.shape[0]
    ## only one target --- cannot be "strain-defining"
    if n_targs == 1:
        continue
    ## basically need sum of targ_col_idxs to sum to something < num rows
    targ_col_sums = anch_calls_filt.iloc[:, targ_col_idxs].sum(axis = 0).tolist()
    less_num_rows = [i for i in range(len(targ_col_sums)) if targ_col_sums[i] < n_targs]
    ## add to file
    if len(less_num_rows) > 0:
        anch_calls_filt.to_csv(strain_defn_spl, mode = 'a', sep = '\t', header = False, index = False)

print('Determining strain-defining anchors for score test...')
uniq_score_anchs = score_calls_filt['anchor'].unique()
targ_col_idxs    = score_calls.columns.str.startswith('targ_align')

for i in tqdm(range(len(uniq_score_anchs))):
    anchor = uniq_score_anchs[i]
    anch_calls_filt = score_calls_filt[score_calls_filt['anchor'] == anchor]
    ## number of targets associated with anchor
    n_targs = anch_calls_filt.shape[0]
    ## only one target --- cannot be "strain-defining"
    if n_targs == 1:
        continue
    ## basically need sum of targ_col_idxs to sum to something < num rows
    targ_col_sums = anch_calls_filt.iloc[:, targ_col_idxs].sum(axis = 0).tolist()
    less_num_rows = [i for i in range(len(targ_col_sums)) if targ_col_sums[i] < n_targs]
    ## add to file
    if len(less_num_rows) > 0:
        anch_calls_filt.to_csv(strain_defn_sc, mode = 'a', sep = '\t', header = False, index = False)



