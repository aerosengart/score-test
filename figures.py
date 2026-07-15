#!/usr/bin/env python3

import argparse
import yaml
import numpy as np
import jax.numpy as jnp

import os

from routines import figures 

def main(): 
    ## create an argument parser and add options
    parser = argparse.ArgumentParser()
    parser.add_argument('-o',
                        '--out_dir', 
                        help = 'Path to save plots.', 
                        type = str, 
                        dest = 'out_dir')

    
    args = parser.parse_args()
    ## check if save path exists
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir, exist_ok = True)
    
    figures.plot_mu_arr(args.out_dir)
    
    


if __name__ == "__main__":
    main()

