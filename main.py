#!/usr/bin/env python3

import argparse
import yaml

from data_module.neg_bin_data import Data as neg_bin_data
from data_module.splash_data import Data as splash_data
from data_module.neg_bin_model import Model as neg_bin_model

from score_module import score, info, intermed
from main_module import wrappers

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

    ## parse the arguments and config file
    args = parser.parse_args()
    with open(args.d_config, 'r') as file:
        d_config = yaml.safe_load(file)
    with open(args.m_config, 'r') as file:
        m_config = yaml.safe_load(file)

    # wrappers.run_routine(args.d_config, args.m_config, 'score_var_comp', reps = 100)
    wrappers.run_routine(args.d_config, args.m_config, 'one_two_sided', iters = 25, reps = 100)

    # d = neg_bin_data(d_config)
    # # d = splash_data(d_config, 'AAAAACAGACACAAGCTCTTATCATGT')
    # m = neg_bin_model(m_config, d)

    # s, sc = score.compute_score_var_comp(m, scale = True)
    # i = info.compute_info(m, sc)
    # print(score.compute_test_stat(m, True))



if __name__ == "__main__":
    main()


