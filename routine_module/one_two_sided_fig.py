"""

"""

import os
import re
import glob

import matplotlib.pyplot as plt
import jax.numpy         as jnp
import numpy             as np

from tqdm        import tqdm
from copy        import deepcopy

from routine_module  import doc
from stat_module import score, helpers, stat

from data_module.dispatch  import data_types
from model_module.dispatch import model_types

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.size"] = 16

def update_n_samps(data, value):
    """Set number of targets."""
    ## resampling is in this function
    data.update_n_samps(value)

def update_n_targs(data, value):
    """Set number of targets."""
    ## resampling is in this function
    data.update_n_targs(value)

param_dispatch = {
    ## go backwards so can remove samples instead of adding more
    'n_samps': [500, 200, 100, 50, 20],
    'n_targs': [1000, 750, 500, 400, 300, 200, 100, 50, 20, 10],
}

update_dispatch = {
    'n_samps': update_n_samps,
    'n_targs': update_n_targs,
}

title_dispatch = {
    'n_samps': 'Number of Samples',
    'n_targs': 'Number of Targets',
}


def one_two_sided_figure(
        d_config,
        m_config,
        param:   str,
        iter_id: int,
        reps:    int = 100,
    ):
    """Compute the one- and two-sided score test statistic for reps number of replicates of Data instance over a grid of values for n_samps (number of biological samples) or n_targs (number of targets).

    The two-sided test statistic is computed in the traditional way where we compute the score, square it, and divide by the corresponding portion of the Fisher information (computed as the negative expected Hessian). The one-sided test statistic is computed as in Silvapulle and Silvapulle.

    ARGS:
        d_config - data configuration dictionary
        m_config - model configuration dictionary
        iter_id  - positive int: iteration number for output name
        reps     - positive int: number of replicates to draw. default: 100
    RAISES:  None
    RETURNS: None
    """
    ## get routine specifics by parameter being varied
    routine = f'one_two_sided_by_{param}'
    value_grid = param_dispatch[param]
    update_param_func = update_dispatch[param]

    ## update seed for iteration
    d_config['seed'] = int(d_config['seed']) + iter_id
    ## update number of samples to be biggest one
    d_config[param] = max(value_grid)
    ## instantiate data
    orig_data = data_types[d_config['outcome']](d_config)
    ## document routine
    out_dir = doc.routine_dir(d_config, m_config, routine)
    ## initialize
    one_sided_stats = jnp.zeros((reps, len(value_grid)))
    two_sided_stats = jnp.zeros((reps, len(value_grid)))

    for r in tqdm(range(reps)):
        ## need to maintain same coefficients and the like
        orig_data.resample()
        data = deepcopy(orig_data)
        data.set_debug(iter = '', rep = str(r), routine = routine, seed = data.seed)
        for i in range(len(value_grid)):
            update_param_func(data, value_grid[i])
            model = model_types[m_config['outcome']](m_config, data)
            ## update debugging dictionary
            if m_config['debug'] == 'full':
                doc.save_rep(model, d_config, m_config, sub_dir = f'{param}{str(value_grid[i])}')
            doc.log_routine(model, d_config, m_config, routine)
            ## compute test statistics
            one_sided_stat, two_sided_stat = stat.compute_score_stat(model, one_sided = True)
            #two_sided_stat = stat.compute_score_stat(model, one_sided = False)
            ## check for crazy large values
            #if one_sided_stat >= 100 or two_sided_stat >= 100:
                #doc.debug_sample(model = model)
            ## record values
            one_sided_stats = one_sided_stats.at[r, i].set(one_sided_stat)
            two_sided_stats = two_sided_stats.at[r, i].set(two_sided_stat)
    ## save iteration
    jnp.savez(
        os.path.join(out_dir, f'results_iter{iter_id}.npz'),
        one_sided_stats = one_sided_stats,
        two_sided_stats = two_sided_stats,
        value_grid      = value_grid
    )
    return out_dir


def process_results(dir_files, out_dir):
    one_stat     = []
    two_stat     = []
    value_grid   = []
    for file in dir_files:
        iter_res = jnp.load(file)
        ## will be the same over all iterations, so we can just overwrite it
        value_grid = iter_res['value_grid']
        one_stat.append(iter_res['one_sided_stats'])
        two_stat.append(iter_res['two_sided_stats'])
    ## save full results
    # jnp.savez(
    #     os.path.join(out_dir, f'results.npz'),
    #     value_grid       = value_grid,
    #     one_sided_stats  = one_stats,
    #     two_sided_stats  = two_stats
    # )
    return value_grid, one_stat, two_stat



def plot_one_two_sided(d_config, m_config, param, reps):
    ## get routine specifics by parameter being varied
    routine = f'one_two_sided_by_{param}'
    ## read all results from output directory 
    out_dir = doc.routine_dir(d_config, m_config, routine)
    dir_files = glob.glob(os.path.join(out_dir, 'results_iter*.npz'))
    ## value_grid is list of ints; one_stats, two_stats are lists of jnp.ararys
    value_grid, one_stats, two_stats = process_results(
        dir_files = dir_files,
        out_dir   = out_dir
    )

    plt_name = os.path.join(out_dir, f'{routine}.pdf')
    plt.rcParams.update({'font.size': 12})
    _, axs = plt.subplots(
        len(value_grid), 2,
        layout = 'constrained',
        figsize = (10, 30)
    )

    ## dims
    q = len(d_config['rd_effs']['sigma'])
    k = d_config['n_targs']
    iters = len(one_stats)
    param_name = title_dispatch[param]

    ## compute theoretical quantiles
    quantiles = np.linspace(0.01, 0.99, 100)
    chisqr_qs = np.vectorize(helpers.chi_sqr_m_quantiles)(quantiles, q)

    ## sample chi bar squared distribution for qq plot since CDF not invertible
    if q > 1:
        chi_bar_sqrs = helpers.chi_bar_sqr_sample(q, k)
        chisqr_mix_qs = np.quantile(chi_bar_sqrs, quantiles)
    ## theoretical quantiles when q = 1
    else:
        chisqr_mix_qs = np.vectorize(helpers.chi_sqr_01_mix_quantiles)(quantiles)

    ## set labels
    axs[len(value_grid) - 1, 0].set_xlabel(r'$\chi^2$ quantiles')
    axs[len(value_grid) - 1, 1].set_xlabel(r'$\bar{\chi}^2$ quantiles')

    ## loop over iterations
    for i in range(iters):
        one_sided = one_stats[i]
        two_sided = two_stats[i]
        for j in range(len(value_grid)):
            val = value_grid[len(value_grid) - (j + 1)]
            one_sided_vals = one_sided[:, len(value_grid) - (j + 1)]
            two_sided_vals = two_sided[:, len(value_grid) - (j + 1)]
            ## compute quantiles
            one_sided_qs = np.quantile(one_sided_vals, quantiles)
            two_sided_qs = np.quantile(two_sided_vals, quantiles)
            ## plot diagonal line and panel titles
            if i == 0:
                axs[j, 0].axline((0, 0), slope = 1, color = 'black', linestyle = 'dotted')
                axs[j, 1].axline((0, 0), slope = 1, color = 'black', linestyle = 'dotted')
                axs[j, 0].set_title(f'Two-Sided: {param_name} = {val}')
                axs[j, 1].set_title(f'One-Sided: {param_name} = {val}')
            ## plot two-sided stats
            axs[j, 0].plot(chisqr_qs, two_sided_qs, color = 'red', alpha = 0.3)
            ## plot one-sided stats
            axs[j, 1].plot(chisqr_mix_qs, one_sided_qs, color = 'blue', alpha = 0.3)
            

    plt.suptitle('QQ Plots of Two-Sided and One-Sided Score Test Statistics')
    plt.savefig(plt_name)
    plt.close()