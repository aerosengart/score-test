#!/usr/bin/env python3

"""
Plot the results of a routine.
"""

import argparse
import yaml
import sys

sys.path.append('/home/arosenga/score-test')
from routine_module import dispatch

def main(): 
    ## create an argument parser and add options
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--d_config', 
                        help = 'Path to data config file.', 
                        type = str, 
                        dest = 'd_config')
    parser.add_argument('-m',
                        '--m_config', 
                        help = 'Path to model config file.', 
                        type = str, 
                        dest = 'm_config')
    parser.add_argument('-r',
                        '--routine', 
                        help = 'Name of routine to run.', 
                        type = str, 
                        dest = 'routine')


    ## parse the arguments and config file
    args = parser.parse_args()

    ## read config files as dictionaries
    with open(args.d_config, 'r') as file:
        d_dict = yaml.safe_load(file)
    with open(args.m_config, 'r') as file:
        m_dict = yaml.safe_load(file)

    ## plot routine
    dispatch.plot_types[args.routine](d_dict, m_dict)

if __name__ == "__main__":
    main()


