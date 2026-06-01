#!/usr/bin/env python3

"""
Repeatedly resample cell counts for a Data instance and compute the component(s) of the score vector corresponding to the random effect(s) variance component(s) to obtain its empirical distribution. 
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
    parser.add_argument('-n',
                        '--reps',
                        help    = 'Number of replicates.',
                        type    = int,
                        dest    = 'reps',
                        default = 100)
    parser.add_argument('-i',
                        '--iter_id',
                        help    = 'Iteration ID.',
                        type    = int,
                        dest    = 'iter_id',
                        default = 1)
    parser.add_argument('-f',
                        '--flag',
                        help    = 'Optional boolean flag specific to routine.',
                        type    = int,
                        dest    = 'flag')

    ## parse the arguments and config file
    args = parser.parse_args()

    ## read config files as dictionaries
    with open(args.d_config, 'r') as file:
        d_dict = yaml.safe_load(file)
    with open(args.m_config, 'r') as file:
        m_dict = yaml.safe_load(file)

    ## run routine (one iteration with ID equal to iter_id in arguments)
    if args.flag is not None:
        ## if we need one more argument for the routine
        dispatch.routine_types[args.routine](
            d_dict, 
            m_dict, 
            args.iter_id,
            args.reps,
            bool(args.flag)
        )
    else:
        dispatch.routine_types[args.routine](
            d_config = d_dict, 
            m_config = m_dict, 
            iter_id  = args.iter_id,
            reps     = args.reps,
        )


if __name__ == "__main__":
    main()


