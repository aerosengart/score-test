#!/usr/bin/env python3

"""
Run the specified routine from configuration file and save results to file.
"""

import jsonargparse
import yaml
import sys
import os

sys.path.append('/home/arosenga/score-test')
from routine_module import dispatch, doc

def main(): 
    ## create an argument parser and add options
    parser = jsonargparse.ArgumentParser()
    parser.add_argument('--cfg', action = jsonargparse.ActionConfigFile)
    parser.add_argument('--d_config', 
                        help = 'Path to data config file.', 
                        dest = 'd_config',
                        type = jsonargparse.typing.Path)
    parser.add_argument('--m_config', 
                        help = 'Path to model config file.', 
                        dest = 'm_config',
                        type = jsonargparse.typing.Path)
    parser.add_argument('--routine', 
                        help = 'Name of routine to run.', 
                        type = str, 
                        dest = 'routine')
    parser.add_argument('--iter_id',
                        help    = 'Iteration ID for logging.',
                        type    = int,
                        dest    = 'iter_id')
    parser.add_argument('--r_args',
                        help    = 'Routine-specific arguments',
                        dest    = 'r_args')

    ## parse the arguments and config file
    args = parser.parse_args()

    ## read config files as dictionaries
    with open(args.d_config, 'r') as file:
        d_dict = yaml.safe_load(file)
    with open(args.m_config, 'r') as file:
        m_dict = yaml.safe_load(file)
    

    # run routine (one iteration with ID equal to iter_id in arguments)
    if args.r_args is not None:
        out_dir = dispatch.routine_types[args.routine](
            d_config = d_dict, 
            m_config = m_dict, 
            iter_id  = args.iter_id,
            **args.r_args
        )
    else:
        out_dir = dispatch.routine_types[args.routine](
            d_config = d_dict, 
            m_config = m_dict, 
            iter_id  = args.iter_id
        )

    ## log configuration files 
    ## only logs to data/model sub-directory, have to manually move to 
    ## routine sub-directory
    doc.log_configs(args.cfg[0], out_dir)

if __name__ == "__main__":
    main()


