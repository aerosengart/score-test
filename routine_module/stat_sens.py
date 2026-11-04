"""
Single iteration routines for assessing test statistic sensitivity to parameter misspecification.

Methods:
    - test_stat_fx_eff_sensitivity
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

##----------------------------------------------------------------------------##
## Wrappers                                                                   ##
##----------------------------------------------------------------------------##
def set_fx_effs_jitter(model, jitter_vals):
    ## jitter the fixed effects
    fx_effs_arr = model.data.fx_effs_arr + jitter_vals
    ## overwrite model MLE with jittered array
    model.set_fx_effs_arr(fx_effs_arr)

def set_disp_jitter(model, jitter_vals):
    ## fully set dispersion (not jitter)
    # if model.data.outcome == 'poi':
    disp_arr = np.ones(model.data.disp_arr.shape) * jitter_vals
    # else:
        # disp_arr = model.data.disp_arr + jitter_vals
    ## overwrite model MLE with jittered array
    model.set_disp_arr(disp_arr)

##----------------------------------------------------------------------------##
## Dispatch Dictionaries                                                      ##
##----------------------------------------------------------------------------##
jitter_dispatch = {
    'fx_effs': set_fx_effs_jitter,
    'disp':    set_disp_jitter,
}

x_axis_vals_dispatch = {
    'fx_effs': [-1, -0.1, -0.05, -0.01, 0.0, 0.01, 0.05, 0.1, 1],
    'disp':    [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 5.0]
}

title_dispatch = {
    'fx_effs': 'QQ Plots of Test Statistics over Jittered Fixed Effects',
    'disp': r'QQ Plots of Test Statistics over Dispersion ($\gamma$)'
}
##----------------------------------------------------------------------------##
## Computations                                                               ##
##----------------------------------------------------------------------------##
def test_stat_sensitivity(
        d_config,
        m_config,
        iter_id:     int,
        param:       str,
        other_known: bool = True,
        reps:        int = 100
    ):
    """
    Compute score test statistic for jittered values of the specified parameter. Optionally, hold the other model parameters fixed at their true values.

    ARGS:
        d_config    - data configuration dictionary
        m_config    - model configuration dictionary
        iter_id     - positive int: iteration number for output name
        param       - str; parameter to jitter (one of 'fx_effs' or 'disp')
        other_known - bool; fix other model parameters at true values
        reps        - int; number of replicates to run per iteration
    RAISES:  None
    RETURNS: None
    """
    ## check parameter to jitter
    if param not in ['fx_effs', 'disp']:
        raise NotImplementedError('Parameter unrecognized.')
    ## check if real data (cannot jitter because no true values)
    if d_config['outcome'] == 'splash':
        raise Exception('Cannot jitter real data')
    
    ## update seed for iteration
    d_config['seed'] = int(d_config['seed']) + iter_id
    ## need to adjust model config so copy
    m_config['use_true'] = other_known
    ## routine name
    if other_known:
        routine = f'test_stat_{param}_sens_others_known'
    else:
        routine = f'test_stat_{param}_sens'
    ## document routine
    out_dir = doc.routine_dir(d_config, m_config, routine)
    ## instantiate data
    data = data_types[d_config['outcome']](d_config)
    ## routine settings
    x_axis_vals = x_axis_vals_dispatch[param]
    jitter_func = jitter_dispatch[param]
    ## initialize
    one_sided = np.zeros((reps, len(x_axis_vals)))
    two_sided = np.zeros((reps, len(x_axis_vals)))

    for r in tqdm(range(reps)):
        ## draw data
        data.set_debug(iter = str(iter_id), rep = str(r), routine = routine, seed = data.seed)
        data.resample()
        ## build model
        model = model_types[m_config['outcome']](m_config, data)
        for k in range(len(x_axis_vals)):
            ## update debugging dictionary
            if m_config['debug'] == 'full':
                doc.save_rep(model, d_config, m_config, sub_dir = f'n_samps_{str(x_axis_vals[k])}')
            ## jitter the appropriate parameterso
            jitter_func(model = model, jitter_vals = x_axis_vals[k])
            ## compute score test
            one_stat, two_stat = stat.compute_score_stat(model, one_sided = True)
            #two_stat = stat.compute_score_stat(model, one_sided = False)
            ## record values
            one_sided[r, k] = one_stat
            two_sided[r, k] = two_stat
            #one_sided = one_sided.at[r, k].set(one_stat)
            #two_sided = two_sided.at[r, k].set(two_stat)
    ## save results
    np.savez(
        os.path.join(out_dir, f'results_iter{iter_id}.npz'),
        x_axis_vals     = x_axis_vals,
        one_sided_stats = one_sided,
        two_sided_stats = two_sided
    )
    return out_dir


##----------------------------------------------------------------------------##
## Plotting                                                                   ##
##----------------------------------------------------------------------------##
def qq_jitter_grid(d_config, m_config, param, reps, other_known):
    """Save grid of QQ-plots with jitter."""
    ## get routine specifics by parameter being varied
    if other_known:
        routine = f'test_stat_{param}_sens_others_known'
    else:
        routine = f'test_stat_{param}_sens'

    ## read all results from output directory
    out_dir = doc.routine_dir(d_config, m_config, routine)
    one_sided = []
    two_sided = []
    x_axis_vals = []
    for file in glob.glob(os.path.join(out_dir, 'results_iter*.npz')):
        iter_res = np.load(file)
        ## overwrite since all the same
        x_axis_vals = iter_res['x_axis_vals']
        ## expand dims because we want to create a 2D array
        one_sided.append(np.expand_dims(iter_res['one_sided_stats'], 1))
        two_sided.append(np.expand_dims(iter_res['two_sided_stats'], 1))

    ## join together results (reps, iters, # jitter vals)
    one_sided_stats = np.concatenate(one_sided, axis = 1)
    two_sided_stats = np.concatenate(two_sided, axis = 1)

    ## get necessary quantities/names
    q = len(d_config['var_comp'])
    k = d_config['n_targs']
    plt_name = os.path.join(out_dir, f'{routine}.pdf')
    iters = len(one_sided)

    plt.rcParams.update({'font.size': 8})
    _, axs = plt.subplots(
        len(x_axis_vals), 2,
        tight_layout = True, 
        figsize = (8, 24)
    )

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

    for i in range(iters):
        ## loop over x-axis ticks
        for l in range(len(x_axis_vals)):
            ## compute quantiles
            one_sided_qs = np.quantile(one_sided_stats[:, i, l], quantiles)
            two_sided_qs = np.quantile(two_sided_stats[:, i, l], quantiles)

            ## plot two-sided stats
            axs[l, 0].axline((0, 0), slope = 1, color = 'black', linestyle = 'dotted')
            axs[l, 0].plot(chisqr_qs, two_sided_qs, color = 'red', alpha = 0.3)
            axs[l, 0].set_ylabel(f'Jitter = {x_axis_vals[l]}')

            ## plot one-sided stats
            axs[l, 1].axline((0, 0), slope = 1, color = 'black', linestyle = 'dotted')
            axs[l, 1].plot(chisqr_mix_qs, one_sided_qs, color = 'blue', alpha = 0.3)

    axs[0, 0].set_title('Two-Sided Score Test')
    axs[0, 1].set_title('One-Sided Score Test')
    axs[len(x_axis_vals) - 1, 0].set_xlabel(r'$\chi^2$ theoretical quantiles')
    axs[len(x_axis_vals) - 1, 1].set_xlabel(r'$\bar{\chi}^2$ theoretical quantiles')
    plt.suptitle(title_dispatch[param])
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(plt_name)
    plt.close()