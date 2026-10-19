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

import subprocess
import jsonargparse
import yaml
import json
import os
import sys
import time

import pandas    as pd
import polars    as pl
import jax.numpy as jnp

from tqdm        import tqdm

sys.path.append('/home/arosenga/score-test')
from stat_module import score, info, helpers

from data_module.dispatch  import data_types
from data_module.helpers   import process_bin
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
    temp_dir = d_dict['paths']['temp_dir']
    satc_dir = d_dict['paths']['satc_dir']

    if not os.path.isdir(out_dir_succ):
        os.makedirs(out_dir_succ, exist_ok = True)
    if not os.path.isdir(out_dir_unst):
        os.makedirs(out_dir_unst, exist_ok = True)
    if not os.path.isdir(out_dir_fail):
        os.makedirs(out_dir_fail, exist_ok = True)
    if not os.path.isdir(temp_dir):
        os.makedirs(temp_dir, exist_ok = True)
    if not os.path.isdir(satc_dir):
        os.makedirs(satc_dir, exist_ok = True)

    fail_file = os.path.join(out_dir_fail, f'failed_anchors_{args.start_idx}_{args.end_idx - 1}.csv')
    with open(fail_file, 'w') as file:
        file.write(f'anchor,bin,idx,error\n')

    ## only read [start_idx, end_idx) rows of list
    ## +1 to start_idx because has_header = False but there _is_ a header (skip_rows will read header on starting line)
    anch_subset = pl.read_csv(args.anch_list, skip_rows = args.start_idx + 1, n_rows = args.end_idx - args.start_idx, has_header = False, separator= ',', new_columns = ['anchor', 'bin', 'first_idx', 'last_idx'])

    ## loop over unique bins
    unique_bins = list(anch_subset['bin'].unique())
    for bin in unique_bins:
        ## list of anchors in bin and (polars!) dataframe of SATC for them
        bin_anchs, bin_df = process_bin(d_dict, bin, anch_subset)
        ## anchors will be consecutive in original list so just need to offset by index of first anchor from this bin
        offset = anch_subset.get_column('anchor').index_of(bin_anchs[0])
        ## have to loop over job anchor list to maintain indexing arithmetic
        for idx, anchor in enumerate(bin_anchs):
            j = idx + args.start_idx + offset
            print(bin, j)

            ## anchor output file
            anchor_succ = os.path.join(out_dir_succ, f'{j}_{anchor}.json')
            anchor_unst = os.path.join(out_dir_unst, f'{j}_{anchor}.json')

            ## anchor score test was already run
            if os.path.isfile(anchor_succ) or os.path.isfile(anchor_unst):
                continue

            ## get anchor table data --- drop anchor column
            anch_table_data = bin_df.filter(pl.col('anchor') == anchor)
            anch_table_data = anch_table_data.drop('anchor')

            ## drop rows with max count < 10
            anch_table_data = anch_table_data.filter(
                    pl.col('count').max().over('target') >= 10
                )

            ## skip anchor if table is too small
            unique_targs = len(anch_table_data.get_column('target').unique())
            unique_samps = len(anch_table_data.get_column('sample').unique())

            if unique_targs < 30 or unique_samps < 200:
                with open(fail_file, 'a') as file:
                    file.write(f'{anchor},{bin},{j},insufficient data\n')
                continue

            ## write table to file
            anch_tab_path = os.path.join(d_dict['paths']['satc_dir'], f'{anchor}.txt')
            anch_table_data.write_csv(anch_tab_path, separator = '\t', include_header = False)

            try: 
                t0 = time.time()
                ## instantiate data and model
                data = data_types[d_dict['outcome']](d_dict, anchor)
                q = data.z_arr.shape[-1]

                ## try to fit model
                try:
                    model = model_types[m_dict['outcome']](m_dict, data)
                except:
                    with open(fail_file, 'a') as file:
                        file.write(f'{anchor},{bin},{j},model fit\n')
                    continue
                ## successful model fit -> into json
                model_info = {
                    'n_samps': model.data.n_samps,
                    'n_targs': model.data.n_targs,
                    'q': q,
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
                ## try to compute score 
                try:
                    ## compute score
                    score_val, scale_val = score.compute_score_eff(
                        model, 
                        scale = model.scale
                    )
                except:
                    with open(fail_file, 'a') as file:
                        file.write(f'{anchor},{bin},{j},score\n')
                    continue
                test_info = {
                    'score_val': score_val.tolist(), 
                    'scale_val': float(scale_val)
                }
                ## try to compute information components
                try:
                    info_mat = info.compute_info_eff(model, scale_val)
                    info_inv = jnp.linalg.pinv(info_mat)
                except:
                    with open(fail_file, 'a') as file:
                        file.write(f'{anchor},{bin},{j},information\n')
                    continue
                test_info['info_inv'] = info_inv.tolist()

                ## try to find infimum
                try:
                    inf_val = helpers.compute_infimum(
                        score     = score_val,
                        info_inv  = info_inv, 
                        test_p    = 0,
                        q         = q
                    )
                except:
                    with open(fail_file, 'a') as file:
                        file.write(f'{anchor},{bin},{j},infimum\n')
                    continue
                
                test_info['inf_val'] = jnp.squeeze(inf_val).tolist()

                ## compute stat
                if q == 1:
                    stat = score_val**2 * info_inv
                else:
                    stat = jnp.matmul(
                        jnp.matmul(jnp.transpose(score_val), info_inv), score_val
                    )
                
                ## try to compute one-sided stat
                try:
                    one_sided_stat = jnp.squeeze(stat - inf_val)
                    test_info['one_side'] = float(one_sided_stat)
                except:
                    with open(fail_file, 'a') as file:
                        file.write(f'{anchor},{bin},{j},one sided stat\n')
                    continue

                ## try to compute two-sided stat
                try:
                    two_sided_stat = jnp.squeeze(stat)
                    test_info['two_side'] = float(two_sided_stat)
                except:
                    with open(fail_file, 'a') as file:
                        file.write(f'{anchor},{bin},{j},two sided stat\n')
                    continue
                ## try to compute one-sided p-value
                try:
                    one_sided_p = helpers.compute_score_test_p_val(one_sided_stat, True)
                except:
                    with open(fail_file, 'a') as file:
                        file.write(f'{anchor},{bin},{j},one sided p-value\n')
                    continue
                test_info['one_side_p'] = one_sided_p

                ## try to compute two-sided p-value
                try:
                    two_sided_p = helpers.compute_score_test_p_val(two_sided_stat, False)
                except:
                    with open(fail_file, 'a') as file:
                        file.write(f'{anchor},{bin},{j},two sided p-value\n')
                    continue
                test_info['two_side_p'] = two_sided_p

                t1 = time.time()
                test_info['time'] = t1 - t0

                ## create json
                anchor_json = {'test':  test_info,
                               'model': model_info}
                
                ## write to file
                if abs(jnp.max(model.fx_effs_arr)) > 1e6:
                    ## unstable results
                    with open(anchor_unst, 'w') as file:
                        json.dump(anchor_json, file, indent = 4)
                else:
                    with open(anchor_succ, 'w') as file:
                        json.dump(anchor_json, file, indent = 4)
            except MemoryError:
                with open(fail_file, 'a') as file:
                    file.write(f'{anchor},{bin},{j},memory error\n')
            except TypeError:
                with open(fail_file, 'a') as file:
                    file.write(f'{anchor},{bin},{j},type error\n')
        
            if d_dict['clean_up']:
                subprocess.call(['rm', anch_tab_path])
            

if __name__ == "__main__":
    main()

