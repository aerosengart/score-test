#!/usr/bin/env python3

import argparse
import yaml
import numpy as np

from data_module.neg_bin_data   import Data  as neg_bin_data
from model_module.neg_bin_model import Model as neg_bin_model
from data_module.poi_data import Data as poi_data

from routine_module.error_comps import error_comparison
from routine_module.stat_comps import score_lrt_comparison
from stat_module import score, info, intermed, stat

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

    data = neg_bin_data(d_config)
    print(data.construct_y())
    data = neg_bin_data(d_config)
    print(data.construct_y())
    # data = poi_data(d_config)
    # model = neg_bin_model(m_config, data)


    # score_lrt_comparison(d_config, m_config, 0, 100)



if __name__ == "__main__":
    main()

