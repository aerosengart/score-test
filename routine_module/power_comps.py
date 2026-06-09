"""
Single iteration routines for assessing test power.

Methods:
    - power_by_var_comp()
    - power_by_n_samps()
    - power_by_n_targs()

"""

import os
import glob

import matplotlib.pyplot as plt
import jax.numpy         as jnp
import numpy             as np

from tqdm        import tqdm

from routine_module  import doc
from score_module import score, helpers

from data_module.dispatch  import data_types
from model_module.dispatch import model_types

## some sort of issue with matplotlib, latex, and the unicode minus sign???
#plt.rcParams['text.usetex'] = True
plt.rcParams['axes.unicode_minus'] = False


##----------------------------------------------------------------------------##
## Computations                                                               ##
##----------------------------------------------------------------------------##
def power_by_var_comp( 
        d_config,
        m_config,
        iter_id: int,
        reps:    int = 100
    ):
    """Plot score function for each component of :math:`\tau^2` to check approximate Gaussianity and zero expectation under the null hypothesis. 

    ARGS:
        d_config - str; path to data configuration .yaml
        m_config - str; path to model configuration .yaml
        iter_id  - int; iteration ID (for labelling)
        reps     - int; number of replicates 
    RAISES:  None
    RETURNS: None
    """
    ## check set-up
    if d_config['rd_slopes'] != 1:
        raise NotImplementedError('Routine only implemented for q = 1.')
    ## initialize
    routine = 'power_by_var_comp'
    var_comps = np.logspace(start = 5, stop = 0.3, base = 0.1, num = 25)

    ## get plot name and document routine
    out_dir = doc.document_routine(d_config, m_config, routine)

    one_stats  = jnp.zeros((reps, len(var_comps)))
    two_stats  = jnp.zeros((reps, len(var_comps)))
    one_p_vals = jnp.zeros((reps, len(var_comps)))
    two_p_vals = jnp.zeros((reps, len(var_comps)))

    for i in tqdm(range(len(var_comps))):
        ## update random effect(s) variance component(s)
        d_config['var_comp'] = [var_comps[i]]
        ## instantiate data
        data = data_types[d_config['outcome']](d_config)
        for r in range(reps):
            data.resample()
            data.set_debug(iter = '', rep = str(i), routine = routine)
            model = model_types[m_config['outcome']](m_config, data)
            ## update debugging dictionary
            if m_config['debug'] == 'full':
                doc.debug_sample(model = model)
            ## compute score test
            one_stat = score.compute_test_stat(model, one_sided = True)
            two_stat = score.compute_test_stat(model, one_sided = False)
            one_p = helpers.compute_score_test_p_val(one_stat, True)
            two_p = helpers.compute_score_test_p_val(two_stat, True)

            one_stats  = one_stats.at[r, i].set(one_stat)
            two_stats  = two_stats.at[r, i].set(two_stat)
            one_p_vals = one_p_vals.at[r, i].set(one_p)
            two_p_vals = two_p_vals.at[r, i].set(two_p)


    ## save full results
    jnp.savez(
        os.path.join(out_dir, routine, f'results_iter{iter_id}.npz'),
        var_comps         = var_comps,
        one_sided_stats  = one_stats,
        two_sided_stats  = two_stats,
        one_sided_p_vals = one_p_vals,
        two_sided_p_vals = two_p_vals
    )


def power_by_n_samps( 
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
    ## check set-up
    if d_config['var_comp'][0] == 0.0:
        raise NotImplementedError('Routine only implemented for non-zero variance components.')
    ## initialize
    routine = 'power_by_n_samps'
    n_samps = [25, 50, 100, 500, 1000]

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


def power_by_n_targs( 
        d_config,
        m_config,
        iter_id: int,
        reps:    int = 100
    ):
    """XXXXX

    ARGS:
        d_config - str; path to data configuration .yaml
        m_config - str; path to model configuration .yaml
        iter_id  - int; iteration ID (for labelling)
        reps     - int; number of replicates 
    RAISES:  None
    RETURNS: None
    """
    ## check set-up
    if d_config['var_comp'][0] == 0.0:
        raise NotImplementedError('Routine only implemented for strictly positive variance components.')
    ## initialize
    routine = 'power_by_n_targs'
    n_targs = [100, 500, 1000, 2000, 5000, 10000]

    ## get plot name and document routine
    out_dir = doc.document_routine(d_config, m_config, routine)
    one_stats  = [None] * len(n_targs)
    two_stats  = [None] * len(n_targs)
    one_p_vals = [None] * len(n_targs)
    two_p_vals = [None] * len(n_targs)

    for i in tqdm(range(len(n_targs))):
        ## update random effect(s) variance component(s)
        d_config['n_targs'] = n_targs[i]
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
        n_targs          = n_targs,
        one_sided_stats  = one_stats,
        two_sided_stats  = two_stats,
        one_sided_p_vals = one_p_vals,
        two_sided_p_vals = two_p_vals
    )


##----------------------------------------------------------------------------##
## Plotting                                                                   ##
##----------------------------------------------------------------------------##

def plot_power_by_var_comp(d_config, m_config):
    """
    Wrapper for one_two_sided_lines() when plotting power analysis over values of var_comp (random effects variance components).
    """
    one_two_sided_lines(
        d_config    = d_config,
        m_config    = m_config,
        x_axis_vals = np.logspace(start = 5, stop = 0.3, base = 0.1, num = 25),
        log_x_axis  = True, 
        routine     = 'power_by_var_comp', 
        x_lab       = r'$\tau^2$', 
        title       = r'Power by $\tau^2$'
    )

def plot_power_by_n_samps(d_config, m_config):
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

def plot_power_by_n_targs(d_config, m_config):
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
    one_p_vals = []
    two_p_vals = []
    for j in range(len(all_iters)):
        iter_res = jnp.load(all_iters[j])
        one_sided_ps = iter_res['one_sided_p_vals']
        two_sided_ps = iter_res['two_sided_p_vals']
        one_p_vals.append(one_sided_ps)
        two_p_vals.append(two_sided_ps)

    ## compute power
    one_p_vals = jnp.concatenate(one_p_vals, axis = 0)
    two_p_vals = jnp.concatenate(two_p_vals, axis = 0)
    one_sigs = one_p_vals <= 0.05
    two_sigs = two_p_vals <= 0.05
    ## sum over iterations
    one_power = jnp.sum(one_sigs, axis = 0) / len(all_iters)
    two_power = jnp.sum(two_sigs, axis = 0) / len(all_iters)

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

