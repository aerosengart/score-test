"""Routines (mostly for plotting)."""

import os
import glob
import jax

from tqdm               import tqdm
from itertools          import repeat

import numpy             as np
import jax.numpy         as jnp
import matplotlib.pyplot as plt

from main_module import plot, dispatch, doc
from score_module import score, helpers


def score_var_comp_empirical_dist(
        d_config,
        m_config,
        reps:   int = 100
    ):
    """Plot score function for each component of :math:`\tau^2` to check approximate Gaussianity and zero expectation under the null hypothesis. 

    ARGS:
        config: 
            a .yaml file specifying the settings for sampling data
        reps:
            integer; number of replicates to run for histogram
    RETURNS:
        None
    RAISES:
        None
    """
    ## instantiate data
    data = dispatch.data_types[d_config['outcome']](d_config)
    ## initialize
    scores_arr = jnp.zeros((data.q, reps)) 
    routine = 'score_var_comp_emp_dist'
    ## get plot name and document routine
    out_dir = doc.document_routine(d_config, m_config, routine)
    ## construct plot name
    plt_name = os.path.join(out_dir, routine, routine + '.pdf')

    for i in tqdm(range(reps)):
        ## draw observations and build model
        data.resample()
        data.set_debug(iter = '', rep = str(i), routine = routine)
        model = dispatch.model_types[m_config['outcome']](m_config, data)
        ## update debugging dictionary
        if m_config['debug'] == 'full':
            doc.debug_sample(model = model)
        ## compute score
        gradient, _ = score.compute_score_var_comp(model, scale = False)
        scores_arr = scores_arr.at[:, i].set(gradient)

    ## save full results
    jnp.savez(
        os.path.join(out_dir, routine, f'results.npz'),
        scores = scores_arr
    )
    ## plot
    plot.score_hist(
        scores_arr = scores_arr, 
        dim        = data.q, 
        plt_name   = plt_name
    )

def one_two_sided_comparison(
        d_config,
        m_config,
        reps:  int = 100,
        iters: int = 1
    ):
    """Compare the one- and two-sided score test statistics.

    The two-sided test statistic is computed in the traditional way where we compute the score, square it, and divide by the corresponding portion of the Fisher information (computed as the negative expected Hessian). The one-sided test statistic is computed as in Silvapulle and Silvapulle.

    ARGS:
        d_config - data configuration dictionary
        m_config - model configuration dictionary
        reps     - positive int: number of replicates to draw. default: 100
        iters    - positive int: number of iterations. results in iters number of lines on the QQ plots. default: 1
    RETURNS:
        None
    RAISES:
        None
    """
    ## instantiate data
    data = dispatch.data_types[d_config['outcome']](d_config)
    routine = 'score_var_comp_emp_dist'
    ## get plot name and document routine
    routine = 'one_two_sided'
    out_dir = doc.document_routine(d_config, m_config, routine)
    ## construct plot name
    plt_name = os.path.join(out_dir, routine, 'one_two_sided.pdf')
    res_name = os.path.join(out_dir, routine, 'results.npz')
    ## initialize
    one_sided_stats = jnp.zeros((reps, iters))
    two_sided_stats = jnp.zeros((reps, iters))

    if not os.path.isfile(res_name):
        for i in range(iters):
            print(f'==============ITERATION {i}==============\n')
            data = dispatch.data_types[d_config['outcome']](d_config)
            for r in tqdm(range(reps)):
                ## draw data and build model
                data.resample()
                data.set_debug(iter = str(i), rep = str(r), routine = routine)
                model = dispatch.model_types[m_config['outcome']](m_config, data)
                ## update debugging dictionary
                if m_config['debug'] == 'full':
                    doc.debug_sample(model = model)
                ## compute test statistics
                one_sided_stat = score.compute_test_stat(model, one_sided = True)
                two_sided_stat = score.compute_test_stat(model, one_sided = False)
                ## record values
                one_sided_stats = one_sided_stats.at[r, i].set(one_sided_stat)
                two_sided_stats = two_sided_stats.at[r, i].set(two_sided_stat)
            ## save each iteration
            jnp.savez(
                os.path.join(out_dir, routine, f'results_iter{i}.npz'),
                one_sided_stats = one_sided_stats,
                two_sided_stats = two_sided_stats
            )
        ## save full results
        jnp.savez(
            os.path.join(out_dir, routine, f'results.npz'),
            one_sided_stats = one_sided_stats,
            two_sided_stats = two_sided_stats
        )

        for filename in glob.glob(os.path.join(
            out_dir, routine, 'results_iter*'
        )):
            os.remove(filename) 
    else:
        results = jnp.load(res_name)
        one_sided_stats = results['one_sided_stats']
        two_sided_stats = results['two_sided_stats']

    plot.one_two_sided_qq(
        one_sided_stats = one_sided_stats, 
        two_sided_stats = two_sided_stats,
        q = data.q,
        k = data.n_targs, 
        plt_name = plt_name
    )
