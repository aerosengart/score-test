#!/usr/bin/env python3

"""
Run (one- AND two-sided) score test based on a negative binomial GLMM on a single anchor S(A)TC table. Output is a .json file in the specified output directory with the following structure:

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
    parser.add_argument('--output',
                        help = 'Output directory.',
                        type = str,
                        dest = 'out_dir')
    parser.add_argument('--anchor', 
                        help = 'Anchor sequence.', 
                        type = str, 
                        dest = 'anchor')
    args = parser.parse_args()

    with open(args.d_config, 'r') as file:
        d_dict = yaml.safe_load(file)

    with open(args.m_config, 'r') as file:
        m_dict = yaml.safe_load(file)

    ## instantiate data and model
    data  = data_types[d_dict['outcome']](d_dict, args.anchor)
    model = model_types[m_dict['outcome']](m_dict, data)

    ## compute all of the pieces for the score test so that we can write to file
    score_val, scale_val = score.compute_score_var_comp(
        model, 
        scale = model.scale
    )
    info_val = info.compute_info(
        model, 
        scale_val = scale_val
    )
    schur_inv = helpers.lower_right_inv(
        mat = info_val,
        q   = data.q
    )
    inf_val = helpers.compute_infimum(
        score     = score_val,
        schur_inv = schur_inv,
        test_p    = 0,
        q         = data.q
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

    model_info = {
        'fx_effs': model.fx_effs_arr.tolist(),
        'disp':   model.disp_arr.tolist()
    }
    test_info = {
        'score_val': float(score_val.tolist()),
        'scale_val': float(scale_val),
        'schur_inv': schur_inv.tolist(),
        'inf_val':   float(inf_val),
        'one_side':  float(one_sided_stat.tolist()),
        'one_side_p': one_sided_p,
        'two_side':  float(two_sided_stat[0][0]),
        'two_side_p': two_sided_p
    }
    anchor_json = {'test':  test_info,
                   'model': model_info}
    
    ## check if save path exists
    out_dir = os.path.join(args.out_dir, 'score_test/anchor_json')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok = True)
    anchor_file = os.path.join(out_dir, f'{args.anchor}.json')
    ## write to file
    with open(anchor_file, 'w') as file:
        json.dump(anchor_json, file, indent = 4)

if __name__ == "__main__":
    main()

