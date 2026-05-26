#!/usr/bin/env python

"""
fastq.py - extract data from SRA accessions in FASTQ format.
REQUIRES: sra_dir - string path to directory of SRA containers
          sra_bin - string path to directory of SRA binaries
MODIFIES: None
RETURNS: None
"""

import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--sra_dir", type = str, help = "path of directory with SRA containers")
parser.add_argument("--sra_bin", type = str, help = "path of directory with SRA binaries")
args = parser.parse_args()

for sra in os.listdir(args.sra_dir):
  if sra.startswith('SR'):
    subprocess.run([os.path.join(args.sra_bin, "fasterq-dump"), "--split-3", os.path.join(args.sra_dir, sra), "-O", os.path.join(args.sra_dir, sra)])
