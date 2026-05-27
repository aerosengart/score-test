#!/usr/bin/env python3

import argparse
import yaml
import json
import os
import sys

import jax.numpy as jnp

sys.path.append('/home/arosenga/score-test')
from score_module import score, info, helpers
from main_module  import dispatch

def main():
    ## create an argument parser, add options, parse arguments
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
    parser.add_argument('-a',
                        '--anchor', 
                        help = 'Anchor sequence.', 
                        type = str, 
                        dest = 'anchor')
    parser.add_argument('-o', 
                        '--output',
                        help = 'Output directory.',
                        type = str,
                        dest = 'out_dir')
    args = parser.parse_args()

    with open(args.d_config, 'r') as file:
        d_dict = yaml.safe_load(file)

    with open(args.m_config, 'r') as file:
        m_dict = yaml.safe_load(file)

    ## instantiate data and model
    data  = dispatch.data_types[d_dict['outcome']](d_dict, args.anchor)
    model = dispatch.model_types[m_dict['outcome']](m_dict, data)

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
        'gamma':   model.gamma.tolist()
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
    out_dir = os.path.join(args.out_dir, 'score_test')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok = True)
    anchor_file = os.path.join(out_dir, f'{args.anchor}.json')
    ## write to file
    with open(anchor_file, 'w') as file:
        json.dump(anchor_json, file, indent = 4)

if __name__ == "__main__":
    main()

