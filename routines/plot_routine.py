#!/usr/bin/env python3

"""
Plot the results of a routine.
"""

import jsonargparse
import yaml
import sys

sys.path.append('/home/arosenga/score-test')
from routine_module import dispatch

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
    ## this is not used, but we need to allow it to be passed so that there is no error thrown and we can re-use same configuration file
    parser.add_argument('--r_args',
                        help    = 'Routine-specific arguments',
                        type    = list,
                        dest    = 'r_args')

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


