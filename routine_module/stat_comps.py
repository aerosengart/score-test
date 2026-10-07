"""
Single iteration routines for comparing statistics of different tests.

Methods:
    - one_two_sided_comparison(); comparing the one- and two-sided score test
"""

import os
import glob

import matplotlib.pyplot as plt
import jax.numpy         as jnp
import numpy             as np

from tqdm        import tqdm

from routine_module  import doc
from stat_module import score, helpers, stat

from data_module.dispatch  import data_types
from model_module.dispatch import model_types

## some sort of issue with matplotlib, latex, and the unicode minus sign???
#plt.rcParams['text.usetex'] = True
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.size"] = 16


##----------------------------------------------------------------------------##
## Computations                                                               ##
##----------------------------------------------------------------------------##
def one_two_sided_comparison(
        d_config,
        m_config,
        iter_id: int,
        reps:    int = 100,
    ):
    """Compute the one- and two-sided score test statistic for reps number of replicates of Data instance.

    The two-sided test statistic is computed in the traditional way where we compute the score, square it, and divide by the corresponding portion of the Fisher information (computed as the negative expected Hessian). The one-sided test statistic is computed as in Silvapulle and Silvapulle.

    ARGS:
        d_config - data configuration dictionary
        m_config - model configuration dictionary
        iter_id  - positive int: iteration number for output name
        reps     - positive int: number of replicates to draw. default: 100
    RAISES:  None
    RETURNS: None
    """
    ## update seed for iteration
    d_config['seed'] = int(d_config['seed']) + iter_id
    ## instantiate data
    data = data_types[d_config['outcome']](d_config)
    ## document routine
    routine = 'one_two_sided'
    out_dir = doc.routine_dir(d_config, m_config, routine)
    ## initialize
    one_sided_stats = jnp.zeros((reps))
    two_sided_stats = jnp.zeros((reps))

    for r in tqdm(range(reps)):
        ## draw data and build model
        data.set_debug(iter = '', rep = str(r), routine = routine, seed = data.seed)
        data.resample()
        model = model_types[m_config['outcome']](m_config, data)
        ## update debugging dictionary
        if m_config['debug'] == 'full':
            doc.save_rep(model, d_config, m_config, sub_dir = None)
        doc.log_routine(model, d_config, m_config, routine)
        ## compute test statistics
        one_sided_stat, two_sided_stat = stat.compute_score_stat(model, one_sided = True)
        #two_sided_stat = stat.compute_score_stat(model, one_sided = False)
        ## check for crazy large values
        #if one_sided_stat >= 100 or two_sided_stat >= 100:
            #doc.debug_sample(model = model)
        ## record values
        one_sided_stats = one_sided_stats.at[r].set(one_sided_stat)
        two_sided_stats = two_sided_stats.at[r].set(two_sided_stat)
    ## save iteration
    jnp.savez(
        os.path.join(out_dir, f'results_iter{iter_id}.npz'),
        one_sided_stats = one_sided_stats,
        two_sided_stats = two_sided_stats
    )
    return out_dir


def score_lrt_comparison(
        d_config,
        m_config,
        iter_id: int,
        reps:    int = 100,
    ):
    """Compute the one-sided score test statistic and LRT statistic for the provided data-generating scheme.

    ARGS:
        d_config - data configuration dictionary
        m_config - model configuration dictionary
        iter_id  - positive int: iteration number for output name
        reps     - positive int: number of replicates to draw. default: 100
    RAISES:  None
    RETURNS: None
    """
    ## instantiate data
    data = data_types[d_config['outcome']](d_config)
    ## document routine
    routine = 'score_lrt_comp'
    out_dir = doc.routine_dir(d_config, m_config, routine)
    ## initialize
    score_stats = np.zeros((reps))
    lrt_stats   = np.zeros((reps))

    for r in tqdm(range(reps)):
        ## draw data and build model
        data.resample()
        data.set_debug(iter = str(iter_id), rep = str(r), routine = routine)
        model = model_types[m_config['outcome']](m_config, data)
        if m_config['debug'] == 'full':
            doc.save_data(data = data, m_config = m_config)
            doc.debug_sample(model = model)
        ## compute test statistics
        score_stat = stat.compute_score_stat(model, one_sided = True)
        lrt_stat   = stat.compute_lrt_stat(model)
        ## check for crazy large values
        if m_config['debug'] == 'full' and (score_stat >= 100 or lrt_stat >= 100):
            doc.debug_sample(model = model)
        ## record values
        score_stats[r] = score_stat
        lrt_stats[r] = lrt_stat
    ## save iteration
    jnp.savez(
        os.path.join(out_dir, f'results_iter{iter_id}.npz'),
        score_stats = score_stats,
        lrt_stats   = lrt_stats
    )
    return out_dir



##----------------------------------------------------------------------------##
## Plotting                                                                   ##
##----------------------------------------------------------------------------##

def plot_one_two_sided(d_config, m_config, reps):
    """
    Plot side-by-side QQ-plots for one- and two-sided score test statistics.

    ARGS:
        d_config - str; path to data configuration .yaml
        m_config - str; path to model configuration .yaml
    RAISES:  None
    RETURNS: None
    """
    ## read all results from output directory
    routine = 'one_two_sided'
    out_dir = doc.routine_dir(d_config, m_config, routine)
    one_sided = []
    two_sided = []
    for file in glob.glob(os.path.join(out_dir, 'results_iter*.npz')):
        iter_res = jnp.load(file)
        ## expand dims because we want to create a 2D array
        one_sided.append(jnp.expand_dims(iter_res['one_sided_stats'], 1))
        two_sided.append(jnp.expand_dims(iter_res['two_sided_stats'], 1))

    ## join together results
    one_sided_stats = jnp.concatenate(one_sided, axis = 1)
    two_sided_stats = jnp.concatenate(two_sided, axis = 1)

    ## get necessary quantities/names
    q = len(d_config['rd_effs']['sigma'])
    k = d_config['n_targs']
    plt_name = os.path.join(out_dir, f'{routine}.pdf')

    plt.rcParams.update({'font.size': 12})
    _, axs = plt.subplots(
        1, 2,
        tight_layout = True, 
        figsize = (12, 6), 
        sharey  = True
    )

    ## set labels
    axs[0].set_xlabel(r'$\chi^2$ quantiles')
    axs[0].set_ylabel('empirical quantiles')
    axs[0].set_title(f'Two-Sided Test Statistic')

    axs[1].set_xlabel(r'$\bar{\chi}^2$ quantiles')
    axs[1].set_ylabel('empirical quantiles')
    axs[1].set_title(f'One-Sided Test Statistic')

    iters = one_sided_stats.shape[-1]

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

    ## loop over columns
    for l in range(iters):
        ## compute quantiles
        one_sided_qs = np.quantile(one_sided_stats[:, l], quantiles)
        two_sided_qs = np.quantile(two_sided_stats[:, l], quantiles)
        
        ## plot two-sided stats
        axs[0].axline((0, 0), slope = 1, color = 'black', linestyle = 'dotted')
        axs[0].plot(chisqr_qs, two_sided_qs, color = 'red', alpha = 0.3)

        ## plot one-sided stats
        axs[1].axline((0, 0), slope = 1, color = 'black', linestyle = 'dotted')
        axs[1].plot(chisqr_mix_qs, one_sided_qs, color = 'blue', alpha = 0.3)

    plt.suptitle('QQ Plots of One-Sided and Two-Sided Test Statistic')
    plt.savefig(plt_name)
    plt.close()




def plot_score_lrt(d_config, m_config, reps):
    """
    Plot side-by-side QQ-plots for LRT statistics and one-sided score test statistics.

    ARGS:
        d_config - str; path to data configuration .yaml
        m_config - str; path to model configuration .yaml
    RAISES:  None
    RETURNS: None
    """
    ## read all results from output directory
    routine = 'score_lrt_comp'
    out_dir = doc.routine_dir(d_config, m_config, routine) 
    score_stats = []
    lrt_stats = []
    for file in glob.glob(os.path.join(out_dir, 'results_iter*.npz')):
        iter_res = jnp.load(file)
        ## expand dims because we want to create a 2D array
        score_stats.append(jnp.expand_dims(iter_res['score_stats'], 1))
        lrt_stats.append(jnp.expand_dims(iter_res['lrt_stats'], 1))

    ## join together results
    score_stats = jnp.concatenate(score_stats, axis = 1)
    lrt_stats = jnp.concatenate(lrt_stats, axis = 1)

    ## get necessary quantities/names
    q = len(d_config['rd_effs']['sigma'])
    k = d_config['n_targs']
    plt_name = os.path.join(out_dir, f'{routine}.pdf')

    plt.rcParams.update({'font.size': 8})
    _, axs = plt.subplots(
        1, 2,
        tight_layout = True, 
        figsize = (12, 6), 
        sharey  = True
    )

    ## set labels
    axs[0].set_xlabel(r'$\bar{\chi}^2$ quantiles')
    axs[0].set_ylabel('empirical quantiles')
    axs[0].set_title(f'Likelihood Ratio Test Statistic')

    axs[1].set_xlabel(r'$\bar{\chi}^2$ quantiles')
    axs[1].set_ylabel('empirical quantiles')
    axs[1].set_title(f'One-Sided Score Test Statistic')

    iters = score_stats.shape[-1]

    ## compute theoretical quantiles
    quantiles = np.linspace(0.01, 0.99, 100)

    ## sample chi bar squared distribution for qq plot since CDF not invertible
    if q > 1:
        chi_bar_sqrs = helpers.chi_bar_sqr_sample(q, k)
        chisqr_mix_qs = np.quantile(chi_bar_sqrs, quantiles)
    ## theoretical quantiles when q = 1
    else:
        chisqr_mix_qs = np.vectorize(helpers.chi_sqr_01_mix_quantiles)(quantiles)

    ## loop over columns
    for l in range(iters):
        ## compute quantiles
        score_qs = np.quantile(score_stats[:, l], quantiles)
        lrt_qs = np.quantile(lrt_stats[:, l], quantiles)
        
        ## plot two-sided stats
        axs[0].axline((0, 0), slope = 1, color = 'black', linestyle = 'dotted')
        axs[0].plot(chisqr_mix_qs, lrt_qs, color = 'red', alpha = 0.3)

        ## plot one-sided stats
        axs[1].axline((0, 0), slope = 1, color = 'black', linestyle = 'dotted')
        axs[1].plot(chisqr_mix_qs, score_qs, color = 'blue', alpha = 0.3)

    plt.suptitle('QQ Plots of Likelihood Ratio Test and One-Sided Score Test Statistics')
    plt.savefig(plt_name)
    plt.close()