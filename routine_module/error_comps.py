"""
Single iteration routines for assessing Type I/II error of tests.

Methods:
    - type_i_error_by_n_samps();
"""

import os
import glob

import matplotlib.pyplot as plt
import jax.numpy         as jnp
import numpy             as np

from tqdm        import tqdm

from routine_module  import doc
from score_module import score, helpers

from data_module.dispatch import data_types, model_types

## some sort of issue with matplotlib, latex, and the unicode minus sign???
#plt.rcParams['text.usetex'] = True
plt.rcParams['axes.unicode_minus'] = False

##----------------------------------------------------------------------------##
## Computations                                                               ##
##----------------------------------------------------------------------------##
def type_i_error_by_n_samps( 
        d_config,
        m_config,
        iter_id: int,
        reps:    int = 100
    ):
    """XXXX

    ARGS:
        d_config - str; path to data configuration .yaml
        m_config - str; path to model configuration .yaml
        iter_id  - int; iteration ID (for labelling)
        reps     - int; number of replicates 
    RAISES:  None
    RETURNS: None
    """
    ## initialize
    routine = 'type_i_error_by_n_samps'
    n_samps = [25, 50, 100, 500, 1000]
    d_config['tau_sqr'] = [0.0] * d_config['rd_slopes']

    ## get plot name and document routine
    out_dir = doc.document_routine(d_config, m_config, routine)
    one_stats  = [None] * len(n_samps)
    two_stats  = [None] * len(n_samps)
    one_p_vals = [None] * len(n_samps)
    two_p_vals = [None] * len(n_samps)

    for i in tqdm(range(len(n_samps))):
        ## update number of samples
        d_config['n_samps'] = n_samps[i]
        ## instantiate data
        data = data_types[d_config['outcome']](d_config)
        data.set_debug(iter = '', rep = str(i), routine = routine)
        model = model_types[m_config['outcome']](m_config, data)
        ## update debugging dictionary
        if m_config['debug'] == 'full':
            doc.debug_sample(model = model)
        ## compute score test
        one_stats[i]  = score.compute_test_stat(model, one_sided = True)
        two_stats[i]  = score.compute_test_stat(model, one_sided = False)
        one_p_vals[i] = helpers.compute_score_test_p_val(one_stats[i], True)
        two_p_vals[i] = helpers.compute_score_test_p_val(two_stats[i], True)

    ## save full results
    jnp.savez(
        os.path.join(out_dir, routine, f'results_iter{iter_id}.npz'),
        n_samps          = n_samps,
        one_sided_stats  = one_stats,
        two_sided_stats  = two_stats,
        one_sided_p_vals = one_p_vals,
        two_sided_p_vals = two_p_vals
    )


##----------------------------------------------------------------------------##
## Plotting                                                                   ##
##----------------------------------------------------------------------------##

