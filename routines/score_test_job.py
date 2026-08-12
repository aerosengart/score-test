#!/usr/bin/env python3

"""
Run (one- AND two-sided) score test based on a negative binomial GLMM on a list of anchor S(A)TC tables. Output per anchor is a .json file in the specified output directory with the following structure:

- test (quantities related to the test itself)
    - score_val: component(s) of score corresponding to the random effect(s) variance component(s)
    - scale_val: value used to scale the score and the information (equal to the 2-norm of the score vector)
    - schur_inv: inverse schur complement of variance of the component(s) of the score corresponding to the random effect(s) variance component(s)
    - inf_val: solution to the quadratic cone program used to construct the one-sided test statistic
    - one_side: one-sided score test statistic
    - one_side_p: one-sided score test p-value (based on chi-bar-squared distribution)
    - two_side: two-sided score test statistic
    - two_side_p: two-sided score test p-value (based on chi-squared distribution)
- model (quantities related to the model/data used in the test)
    - fx_effs: estimated fixed effects coefficients
    - disp: estimated target-specific negative binomial dispersion parameters
"""

import jsonargparse
import yaml
import json
import os
import sys

import jax.numpy as jnp

sys.path.append('/home/arosenga/score-test')
from stat_module import score, info, helpers

from data_module.dispatch  import data_types
from model_module.dispatch import model_types

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
    parser.add_argument('--anchor_list',
                        help = 'Anchor list file.',
                        type = str,
                        dest = 'anch_list')
    parser.add_argument('--output',
                        help = 'Output directory.',
                        type = str,
                        dest = 'out_dir')
    parser.add_argument('--start_idx',
                        help = 'Starting anchor index.',
                        type = int,
                        dest = 'start_idx')
    parser.add_argument('--end_idx',
                        help = 'Ending anchor index.',
                        type = int,
                        dest = 'end_idx')
    args = parser.parse_args()

    ## same data and model configuration files so can load just one time
    with open(args.d_config, 'r') as file:
        d_dict = yaml.safe_load(file)
    with open(args.m_config, 'r') as file:
        m_dict = yaml.safe_load(file)

    ## check if save path exists just one time
    out_dir_succ = os.path.join(m_dict['out_dir'], 'score_test/success')
    out_dir_unst = os.path.join(m_dict['out_dir'], 'score_test/unstable')
    out_dir_fail = os.path.join(m_dict['out_dir'], 'score_test/failure')
    if not os.path.isdir(out_dir_succ):
        os.makedirs(out_dir_succ, exist_ok = True)
    if not os.path.isdir(out_dir_unst):
        os.makedirs(out_dir_unst, exist_ok = True)
    if not os.path.isdir(out_dir_fail):
        os.makedirs(out_dir_fail, exist_ok = True)
    
    ## load specified rows of anchor_list
    anch_seqs = []
    with open(args.anch_list) as fp:
        for i, line in enumerate(fp):
            idx = i + 1
            if (idx >= args.start_idx) and (idx <= args.end_idx):
                anch_seqs += [line.strip()]
            elif idx < args.start_idx:
                continue
            else: 
                break

    ## loop over specified anchors
    for anchor in anch_seqs:
        ## anchor output file
        anchor_succ = os.path.join(out_dir_succ, f'{anchor}.json')
        anchor_unst = os.path.join(out_dir_unst, f'{anchor}.json')
        anchor_fail = os.path.join(out_dir_fail, f'{anchor}.csv')
        ## instantiate data and model
        data = data_types[d_dict['outcome']](d_dict, anchor)

        try:
            model = model_types[m_dict['outcome']](m_dict, data)
        except:
            with open(anchor_fail, 'w') as file:
                file.write(f'{anchor},{data.n_samps},{data.n_targs}, model fitting error')
            continue

        try:
            ## compute all of the pieces for the score test so that we can write to file
            score_val, scale_val = score.compute_score_var_comp(
                model, 
                scale = model.scale
            )
            info_fx, info_fx_var, info_var = info.compute_info(
                model, 
                scale_val = scale_val
            )
            schur_inv = helpers.lower_right_inv(
                upp_left  = info_fx,
                middle    = info_fx_var,
                low_right = info_var
            )
            inf_val = helpers.compute_infimum(
                score = score_val,
                schur_inv = schur_inv,
                test_p = 0,
                q = model.data.q
            )

            ## compute stat
            if model.data.q == 1:
                stat = score_val**2 * schur_inv
            else:
                stat = jnp.matmul(
                    jnp.matmul(jnp.transpose(score_val), schur_inv), score_val
                )

            one_sided_stat = jnp.squeeze(stat - inf_val)
            two_sided_stat = stat
            ## compute p-values
            one_sided_p = helpers.compute_score_test_p_val(one_sided_stat, True)
            two_sided_p = helpers.compute_score_test_p_val(two_sided_stat[0][0], False)
        except:
            with open(anchor_fail, 'w') as file:
                file.write(f'{anchor},{data.n_samps},{data.n_targs}, score test computation error')
            continue
        else:
            ## if we get here, everything went smoothly and we can record the results
            ## summary statistics taken over targets (e.g. min_fx_effs should have p entries)
            model_info = {
                'n_samps': model.data.n_samps,
                'n_targs': model.data.n_targs,
                'q': model.data.q,
                'x_dims': model.data.x_arr.shape,
                'z_dims': model.data.z_arr.shape,
                'fx_effs_dims': model.fx_effs_arr.shape,
                'min_fx_effs': jnp.min(model.fx_effs_arr, axis = 0).tolist(), 
                'max_fx_effs': jnp.max(model.fx_effs_arr, axis = 0).tolist(),
                'med_fx_effs': jnp.median(model.fx_effs_arr, axis = 0).tolist(),
                'min_disp': jnp.min(model.disp_arr, axis = 0).tolist(),
                'max_disp': jnp.max(model.disp_arr, axis = 0).tolist(),
                'med_disp': jnp.median(model.disp_arr, axis = 0).tolist(),
            }
            test_info = {
                'score_val': score_val.tolist(), 
                'scale_val': float(scale_val),
                'schur_inv': schur_inv.tolist(),
                'inf_val':   float(inf_val),
                'one_side':  float(one_sided_stat),
                'one_side_p': one_sided_p,
                'two_side':  float(two_sided_stat[0][0]),
                'two_side_p': two_sided_p
            }
            anchor_json = {'test':  test_info,
                           'model': model_info}
            
            ## write to file
            if abs(jnp.max(model.fx_effs_arr)) > 1e9:
                ## unstable results
                with open(anchor_unst, 'w') as file:
                    json.dump(anchor_json, file, indent = 4)
            else:
                with open(anchor_succ, 'w') as file:
                    json.dump(anchor_json, file, indent = 4)
        
            

if __name__ == "__main__":
    main()

