import os
import glob

import matplotlib.pyplot as plt
import numpy             as np
import jax.numpy         as jnp

from score_module import helpers
from routine_module  import doc


## some sort of issue with matplotlib, latex, and the unicode minus sign???
#plt.rcParams['text.usetex'] = True
plt.rcParams['axes.unicode_minus'] = False


def score_hist(d_config, m_config):
    """
    Plot empirical distribution of component(s) of score vector.

    ARGS:
        d_config - str; path to data configuration .yaml
        m_config - str; path to model configuration .yaml
    RAISES:  None
    RETURNS: None
    """
    ## read all results from output directory
    routine = 'score_var_comp_emp_dist'
    out_dir = doc.output_directory_path(d_config, m_config)
    results = []
    for file in glob.glob(os.path.join(out_dir, routine, 'results_iter*.npz')):
        iter_res = jnp.load(file)
        results.append(iter_res['scores'])

    ## join together results
    scores_arr = jnp.concatenate(results)

    ## get necessary quantities/names
    dim = len(d_config['tau_sqr'])
    plt_name = os.path.join(out_dir, routine, 'score_var_comp.pdf')

    if dim == 1:
        ## variance component is scalar-valued
        _, axs = plt.subplots(
            1, 1, 
            tight_layout = True,
            figsize = (10, 6)
        )
        axs.hist(scores_arr.tolist(), bins = 100, density = True)    
        axs.set_title(f'analytical score component of $\tau^2$ under $H_0$')
        axs.set_xlabel('score')
        axs.set_ylabel('density')

    else:
        _, axs = plt.subplots(
            dim, 1, 
            tight_layout = True,
            figsize = (10, dim * 2 + 4)
        )
        for i in range(dim):
            axs[i].hist(scores_arr[i, :], bins = 100, density = True)    
            axs[i].set_title(f'analytical score component of $\tau^2_{{{i}}}$ under $H_0$')
            axs[i].set_xlabel('score')
            axs[i].set_ylabel('density')
    plt.savefig(plt_name)
    plt.close()






def power_by_tau_sqr(d_config, m_config):
    """
    Wrapper for one_two_sided_lines() when plotting power analysis over values of tau_sqr (random effects variance components).
    """
    one_two_sided_lines(
        d_config    = d_config,
        m_config    = m_config,
        x_axis_vals = np.logspace(start = 5, stop = 0.3, base = 0.1, num = 25),
        log_x_axis  = True, 
        routine     = 'power_by_var_comp', 
        x_lab       = f'$\tau^2$', 
        title       = f'Power by $\tau^2$'
    )

def power_by_n_samps(d_config, m_config):
    """
    Wrapper for one_two_sided_lines() when plotting power analysis over different numbers of samples.
    """
    one_two_sided_lines(
        d_config    = d_config,
        m_config    = m_config,
        x_axis_vals = [25, 50, 100, 500, 1000],
        log_x_axis  = None, 
        routine     = 'power_by_n_samps', 
        x_lab       = 'Number of Samples', 
        title       = f'Power by Sample Size'
    )

def power_by_n_targs(d_config, m_config):
    """
    Wrapper for one_two_sided_lines() when plotting power analysis over different numbers of targets.
    """
    one_two_sided_lines(
        d_config    = d_config,
        m_config    = m_config,
        x_axis_vals = [100, 500, 1000, 2000, 5000, 10000],
        log_x_axis  = 100, 
        routine     = 'power_by_n_targs', 
        x_lab       = 'Number of Targets', 
        title       = f'Power by Number of Targets'
    )



def one_two_sided_lines(
        d_config, m_config, x_axis_vals, log_x_axis, routine, x_lab, title,
    ):
    """
    Plot side-by-side QQ-plots for one- and two-sided score test statistics.

    ARGS:
        d_config - str; path to data configuration .yaml
        m_config - str; path to model configuration .yaml
    RAISES:  None
    RETURNS: None
    """
    ## read all results from output directory
    out_dir = doc.output_directory_path(d_config, m_config)
    all_iters = glob.glob(os.path.join(out_dir, routine, 'results_iter*.npz'))
    one_p_vals = jnp.zeros((len(x_axis_vals), len(all_iters)))
    two_p_vals = jnp.zeros((len(x_axis_vals), len(all_iters)))
    for j in range(len(all_iters)):
        iter_res = jnp.load(all_iters[j])
        one_sided_ps = iter_res['one_sided_p_vals']
        two_sided_ps = iter_res['two_sided_p_vals']
        one_p_vals = one_p_vals.at[:, j].set(one_sided_ps)
        two_p_vals = two_p_vals.at[:, j].set(two_sided_ps)

    ## compute power
    one_sigs = one_p_vals <= 0.05
    two_sigs = two_p_vals <= 0.05
    one_power = jnp.sum(one_sigs, axis = 1) / len(all_iters)
    two_power = jnp.sum(two_sigs, axis = 1) / len(all_iters)

    ## get necessary quantities/names
    plt_name = os.path.join(out_dir, routine, f'{routine}.pdf')
    plt.rcParams.update({'font.size': 10})
    _, axs = plt.subplots(
        1, 1,
        tight_layout = True, 
        figsize = (8, 8)
    )

    ## set labels
    if log_x_axis is not None:
        axs.semilogx(log_x_axis)
    axs.set_xlabel(x_lab)
    axs.set_ylabel('Power')
    axs.set_title(title)

    ## plot
    axs.plot(x_axis_vals, one_power, color = 'blue', label = 'one-sided')
    axs.plot(x_axis_vals, two_power, color = 'red',  label = 'two-sided')
    axs.legend()
    ## save
    plt.savefig(plt_name)
    plt.close()








    # ## plot
    # plot.lrt_score_jitter_grid(
    #     lrt_stats = lrt_stats, 
    #     score_stats = score_stats, 
    #     q = data.q,
    #     k = data.k, 
    #     x_axis_ticks = x_axis_ticks, 
    #     plt_name = plt_name
    # )

def fx_effs_jitter(d_config, m_config):
    qq_jitter_grid(d_config, m_config, jitter_vals = [-0.1, -0.05, -0.025, -0.01, 0.0, 0.01, 0.025, 0.05, 0.1], routine = 'test_stat_fx_effs_sens_disp_known')    


def qq_jitter_grid(d_config, m_config, jitter_vals, routine):
    """Save grid of QQ-plots with jitter."""
    ## read all results from output directory
    out_dir = doc.output_directory_path(d_config, m_config)
    one_sided = []
    two_sided = []
    for file in glob.glob(os.path.join(out_dir, routine, 'results_iter*.npz')):
        iter_res = jnp.load(file)
        ## expand dims because we want to create a 2D array
        one_sided.append(jnp.expand_dims(iter_res['one_sided_stats'], 1))
        two_sided.append(jnp.expand_dims(iter_res['two_sided_stats'], 1))

    ## join together results (reps, iters, # jitter vals)
    one_sided_stats = jnp.concatenate(one_sided, axis = 1)
    two_sided_stats = jnp.concatenate(two_sided, axis = 1)

    ## get necessary quantities/names
    q = len(d_config['tau_sqr'])
    k = d_config['n_targs']
    plt_name = os.path.join(out_dir, routine, f'{routine}.pdf')
    iters = len(one_sided)

    plt.rcParams.update({'font.size': 8})
    _, axs = plt.subplots(
        len(jitter_vals), 2,
        tight_layout = True, 
        figsize = (8, 24)
    )

    ## compute theoretical quantiles
    quantiles = np.linspace(0.01, 0.99, 100)
    ## sample chi bar squared distribution for qq plot since CDF not invertible
    if q > 1:
        chi_bar_sqrs = helpers.chi_bar_sqr_sample(q, k)
        chisqr_mix_qs = np.quantile(chi_bar_sqrs, quantiles)
    ## theoretical quantiles when q = 1
    else:
        chisqr_mix_qs = np.vectorize(helpers.chi_sqr_01_mix_quantiles)(quantiles)

    for i in range(iters):
        ## loop over x-axis ticks
        for l in range(len(jitter_vals)):
            ## compute quantiles
            lrt_qs   = np.quantile(one_sided_stats[:, i, l], quantiles)
            score_qs = np.quantile(two_sided_stats[:, i, l], quantiles)
            
            ## plot two-sided stats
            axs[l, 0].axline((0, 0), slope = 1, color = 'black', linestyle = 'dotted')
            axs[l, 0].plot(chisqr_mix_qs, lrt_qs, color = 'red', alpha = 0.3)
            axs[l, 0].set_ylabel(f'Jitter = {jitter_vals[l]}')

            ## plot one-sided stats
            axs[l, 1].axline((0, 0), slope = 1, color = 'black', linestyle = 'dotted')
            axs[l, 1].plot(chisqr_mix_qs, score_qs, color = 'blue', alpha = 0.3)

    axs[0, 0].set_title('Likelihood Ratio Test')
    axs[0, 1].set_title('Score Test')
    axs[len(jitter_vals) - 1, 0].set_xlabel(r'$\bar{\chi}^2$ theoretical quantiles')
    axs[len(jitter_vals) - 1, 1].set_xlabel(r'$\bar{\chi}^2$ theoretical quantiles')
    plt.suptitle('QQ Plots of One-Sided Test Statistics with Jitter')
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(plt_name)
    plt.close()