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
from score_module import score, helpers

from data_module.dispatch import data_types, model_types

## some sort of issue with matplotlib, latex, and the unicode minus sign???
#plt.rcParams['text.usetex'] = True
plt.rcParams['axes.unicode_minus'] = False

##----------------------------------------------------------------------------##
## Computations                                                               ##
##----------------------------------------------------------------------------##


def set_fx_effs_jitter(model, jitter_vals):
    ## jitter the fixed effects
    fx_effs_arr = model.data.fx_effs_arr + jitter_vals
    ## overwrite model MLE with jittered array
    model.set_fx_effs_arr(fx_effs_arr)
    model.update_params()


def set_disp_jitter(model, jitter_vals):
    ## jitter the dispersion
    disp_arr = model.data.gamma + jitter_vals
    ## overwrite model MLE with jittered array
    model.set_disp(disp_arr)
    model.update_params()
    

def test_stat_sensitivity(
        d_config,
        m_config,
        param:       str,
        iter_id:     int,
        reps:        int = 100,
        other_known: bool = True,
    ):
    """
    Compute score test statistic for jittered values of the specified parameter. Optionally, hold the other model parameters fixed at their true values.

    ARGS:
        d_config    - data configuration dictionary
        m_config    - model configuration dictionary
        param       - str; parameter to jitter (one of 'fx_effs' or 'disp')
        iter_id     - positive int: iteration number for output name
        reps        - int; number of replicates to run per iteration
        other_known - bool; fix other model parameters at true values
    RAISES:  None
    RETURNS: None
    """
    ## check if real data (cannot jitter because no true values)
    if d_config['outcome'] == 'splash':
        raise Exception('Cannot jitter real data')

    ## need to adjust model config so copy
    new_config = m_config

    ## routine name
    if other_known:
        new_config['use_true'] = True
        routine = f'test_stat_{param}_sens_others_known'
    else:
        new_config['use_true'] = False
        routine = f'test_stat_{param}_sens'

    ## document routine
    out_dir = doc.document_routine(d_config, new_config, routine)

    ## instantiate data
    data = data_types[d_config['outcome']](d_config)

    ## initialize
    if param == 'fx_effs':
        x_axis_ticks = [-1, -0.1, -0.05, -0.01, 0.0, \
                        0.01, 0.05, 0.1, 1]
    elif param == 'disp':
        ## set dispersion parameter to true values
        new_config['global_disp'] = data.disp_arr.ravel().tolist()
        x_axis_ticks = [-0.5, -0.25, -0.1, -0.05, 0.0, \
                        0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    else:
        raise NotImplementedError('Parameter unrecognized.')
    
    one_sided = jnp.zeros((reps, len(x_axis_ticks)))
    two_sided = jnp.zeros((reps, len(x_axis_ticks)))

    for r in tqdm(range(reps)):
        ## draw data
        data.resample()
        data.set_debug(iter = str(iter_id), rep = str(r), routine = routine)
        ## build model
        model = model_types[new_config['outcome']](new_config, data)
        ## update debugging dictionary
        if new_config['debug'] == 'full':
            doc.debug_sample(model = model)

        for k in range(len(x_axis_ticks)):
            ## jitter the fixed effects
            fx_effs_arr = data.fx_effs_arr + x_axis_ticks[k]
            ## overwrite model MLE with jittered array
            model.set_fx_effs_arr(fx_effs_arr)
            model.update_params()
            ## compute score test
            one_stat = score.compute_test_stat(model, one_sided = True)
            two_stat = score.compute_test_stat(model, one_sided = False)
            ## record values
            one_sided = one_sided.at[r, k].set(one_stat)
            two_sided = two_sided.at[r, k].set(two_stat)
    ## save results
    jnp.savez(
        os.path.join(out_dir, routine, f'results_iter{iter_id}.npz'),
        one_sided_stats = one_sided,
        two_sided_stats = two_sided
    )



def test_stat_gamma_sensitivity(
        d_config,
        m_config,
        iter_id:       int,
        reps:          int = 100,
        fx_effs_known: bool = True,
    ):
    """
    Compute score test statistic for jittered values of the negative binomial dispersion parameter(s). Optionally, hold fixed effects coefficients fixed at the true values.

    ARGS:
        d_config      - data configuration dictionary
        m_config      - model configuration dictionary
        iter_id       - positive int: iteration number for output name
        reps          - int; number of replicates to run per iteration
        fx_effs_known - bool; fix fixed effects coefficients at true value
    RETURNS:
        None
    RAISES:
        None
    """
    if d_config['outcome'] == 'splash':
        raise Exception('Cannot jitter real data')
    
    ## don't estimate because we are setting the fixed effects directly
    new_config = m_config

    ## routine name
    if fx_effs_known:
        new_config['use_true'] = True
        routine = 'test_stat_gamma_sens_fx_effs_known'
    else:
        new_config['use_true'] = False
        routine = 'test_stat_gamma_sens'

    ## initialize
    x_axis_ticks = [-0.5, -0.25, -0.1, -0.05, 0.0, \
                    0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    
    ## document routine
    out_dir = doc.document_routine(d_config, new_config, routine)

    ## instantiate data
    data = data_types[d_config['outcome']](d_config)

    ## construct plot name
    plt_name = os.path.join(out_dir, routine, routine) + '.pdf'
    one_sided = jnp.zeros((reps, len(x_axis_ticks)))
    two_sided = jnp.zeros((reps, len(x_axis_ticks)))

    for r in tqdm(range(reps)):
        ## draw data
        data.resample()
        data.set_debug(iter = str(iter_id), rep = str(r), routine = routine)
        ## build model
        model = model_types[new_config['outcome']](new_config, data)
        ## update debugging dictionary
        if new_config['debug'] == 'full':
            doc.debug_sample(model = model)

        for k in range(len(x_axis_ticks)):
            ## jitter the fixed effects
            fx_effs_arr = data.fx_effs_arr + x_axis_ticks[k]
            ## overwrite model MLE with jittered array
            model.set_fx_effs_arr(fx_effs_arr)
            model.update_params()
            ## compute score test
            one_stat = score.compute_test_stat(model, one_sided = True)
            two_stat = score.compute_test_stat(model, one_sided = False)
            ## record values
            one_sided = one_sided.at[r, k].set(one_stat)
            two_sided = two_sided.at[r, k].set(two_stat)
    ## save results
    jnp.savez(
        os.path.join(out_dir, routine, f'results_iter{iter_id}.npz'),
        one_sided_stats = one_sided,
        two_sided_stats = two_sided
    )

##----------------------------------------------------------------------------##
## Plotting                                                                   ##
##----------------------------------------------------------------------------##

def gamma_jitter(d_config, m_config):
    qq_jitter_grid(d_config, m_config, jitter_vals = [-0.5, -0.25, -0.1, -0.05, 0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0], routine = 'test_stat_gamma_sens_fx_effs_known')    

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
        for l in range(len(jitter_vals)):
            ## compute quantiles
            one_sided_qs = np.quantile(one_sided_stats[:, i, l], quantiles)
            two_sided_qs = np.quantile(two_sided_stats[:, i, l], quantiles)
            
            ## plot two-sided stats
            axs[l, 0].axline((0, 0), slope = 1, color = 'black', linestyle = 'dotted')
            axs[l, 0].plot(chisqr_qs, two_sided_qs, color = 'red', alpha = 0.3)
            axs[l, 0].set_ylabel(f'Jitter = {jitter_vals[l]}')

            ## plot one-sided stats
            axs[l, 1].axline((0, 0), slope = 1, color = 'black', linestyle = 'dotted')
            axs[l, 1].plot(chisqr_mix_qs, one_sided_qs, color = 'blue', alpha = 0.3)

    axs[0, 0].set_title('Two-Sided Score Test')
    axs[0, 1].set_title('One-Sided Score Test')
    axs[len(jitter_vals) - 1, 0].set_xlabel(r'$\chi^2$ theoretical quantiles')
    axs[len(jitter_vals) - 1, 1].set_xlabel(r'$\bar{\chi}^2$ theoretical quantiles')
    plt.suptitle('QQ Plots of Test Statistics with Jitter')
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(plt_name)
    plt.close()