#!/usr/bin/env python3

import argparse
import yaml
import numpy as np
import jax.numpy as jnp

from data_module import debug

from data_module.dispatch import data_types
from model_module.dispatch import model_types
from routine_module.error_comps import error_comparison, plot_grid_error_comp
from routine_module.stat_comps import score_lrt_comparison
from stat_module import score, info, intermed, stat, helpers

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


    # path = '/home/arosenga/score-test/nb_sims/by_n_samps/type_ii/rd_effs_sigma_0.05/no_batch/fixed_fx_rd_time_reg'

    # plot_grid_error_comp(path, err_type = 'II', param = 'n_samps', int_only = True)

    # path = '/home/arosenga/score-test/nb_sims/by_n_samps/type_i/no_batch/varied_fx_rd_time_reg_use_true2/dneg_bin_mneg_bin_fx_p1_rd_q1_samps50_targs500/type_I_error_by_n_samps/iter/n_samps50/rep0_model.pickle'

    # debug.visualize_data(path)

    data = data_types[d_config['outcome']](d_config, 'GATTCAACAATTTGTTTGAATGCTTTATAAT')
    print(data.n_targs)
    print(data.samples)
    # model = model_types[m_config['outcome']](m_config, data)


    
    # ## compute score
    # score_val, scale_val = score.compute_score_var_comp(
    #     model, 
    #     scale = model.scale
    # )
    # ## compute information
    # info_fx, info_fx_var, info_var = info.compute_info(
    #     model, 
    #     scale_val = scale_val
    # )
    # ## schur complement
    # schur_inv = helpers.lower_right_inv(
    #     upp_left  = info_fx,
    #     middle    = info_fx_var,
    #     low_right = info_var
    # )
    # print(score_val, schur_inv)

    # inf_val = helpers.compute_infimum(
    #     score = score_val,
    #     schur_inv = schur_inv,
    #     test_p = 0,
    #     q = 1
    # )
    # print(inf_val)

    # st = stat.compute_score_stat(model, True)
    # print(st)

    # m_config['use_true'] = True
    # model = neg_bin_model(m_config, data)
    # score_val, scale_val = score.compute_score_var_comp(
    #     model, 
    #     scale = model.scale
    # )
    # ## compute information
    # info_fx, info_fx_var, info_var = info.compute_info(
    #     model, 
    #     scale_val = scale_val
    # )
    # ## schur complement
    # schur_inv = helpers.lower_right_inv(
    #     upp_left  = info_fx,
    #     middle    = info_fx_var,
    #     low_right = info_var
    # )
    # print(score_val, schur_inv)


    # inf_val = helpers.compute_infimum(
    #     score = score_val,
    #     schur_inv = schur_inv,
    #     test_p = 0,
    #     q = 1
    # )
    # print(inf_val)
    
    # st = stat.compute_score_stat(model, True)
    # print(st)


    # ## compute information
    # info_fx, info_fx_var, info_var = info.compute_info(
    #     model, 
    #     scale_val = scale_val
    # )
    # ## schur complement
    # schur_inv = helpers.lower_right_inv(
    #     upp_left  = info_fx,
    #     middle    = info_fx_var,
    #     low_right = info_var
    # )



    # print(stat)

    # # score_lrt_comparison(d_config, m_config, 0, 100)



if __name__ == "__main__":
    main()

