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
from stat_module import score, helpers, stat
from oasis_module import oasis

from data_module.dispatch  import data_types
from model_module.dispatch import model_types

## some sort of issue with matplotlib, latex, and the unicode minus sign???
#plt.rcParams['text.usetex'] = True
plt.rcParams['axes.unicode_minus'] = False

##----------------------------------------------------------------------------##
## Helpers                                                                    ##
##----------------------------------------------------------------------------##
def update_disp(data, value):
    """Set dispersion parameter value."""
    data.set_disp(jnp.ones(data.disp_arr.shape) * value)
    data.resample()

def update_n_samps(data, value):
    """Set number of targets."""
    data.update_n_samps(value)


def update_n_targs(data, value):
    """Set number of targets."""
    data.update_n_targs(value)


def compute_p_vals_intercept_only(data, model):
    """Compute p-values for all tests (OASIS original, OASIS asymptotic, one-sided score test, two-sided score test) when null model has NO fixed slopes."""
    orig, _, _ = oasis.oasis(data, asymptotic = False)
    asymp, _, _ = oasis.oasis(data, asymptotic = True)
    ## compute score test
    one_stat = stat.compute_score_stat(model, one_sided = True)
    two_stat  = stat.compute_score_stat(model, one_sided = False)
    one_side = helpers.compute_score_test_p_val(one_stat, one_sided = True)
    two_side = helpers.compute_score_test_p_val(two_stat, one_sided = False)
    return float(orig), float(asymp), one_side, two_side

def compute_p_vals(data, model):
    """Compute p-values for all tests (OASIS-perp, one-sided score test, two-sided score test) when null model has fixed slopes."""
    ## compute oasis --- default settings!!!
    perp, _, _ = oasis.oasis_perp(data)
    ## compute score test
    one_stat = stat.compute_score_stat(model, one_sided = True)
    two_stat  = stat.compute_score_stat(model, one_sided = False)
    one_side = helpers.compute_score_test_p_val(one_stat, one_sided = True)
    two_side = helpers.compute_score_test_p_val(two_stat, one_sided = False)
    return perp, None, one_side, two_side

def compute_type_i_err(p_vals_arr, level, reps):
    """Reject H0 when true."""
    return (p_vals_arr <= level).sum(axis = 0) / reps

def compute_type_ii_err(p_vals_arr, level, reps):
    """Fail to reject H0 when not true."""
    return (p_vals_arr > level).sum(axis = 0) / reps

def check_type_i_settings(d_config, m_config, param):
    """Check that the data simulation scheme is appropriate for Type I"""
    if len(d_config['z']['types']) != 1:
        raise NotImplementedError('Routine only implemented for q = 1.')
    if len(d_config['x']['types']) > 1:
        raise NotImplementedError('Routine only implemented for p = 1 or 2.')
    if d_config['var_comp'][0] != 0.0:
        raise NotImplementedError('Data must be generated under H0.')
    if param not in ['n_targs', 'disp']:
        raise NotImplementedError('Varied parameter not implemented.')
    if isinstance(m_config['global_disp'], list) and not m_config['use_true']:
        raise NotImplementedError('Model must estimate dispersion or use true values for all parameters.')

def check_type_ii_settings(d_config, m_config, param):
    """Check that the data simulation scheme is appropriate for Type II."""
    if len(d_config['z']['types']) != 1:
        raise NotImplementedError('Routine only implemented for q = 1.')
    if len(d_config['x']['types']) > 1:
        raise NotImplementedError('Routine only implemented for p = 1 or 2.')
    if d_config['var_comp'][0] == 0.0 and d_config['rd_effs']['mu'] == 0.0:
        raise NotImplementedError('Data must be generated under H1.')
    if param not in ['n_targs', 'disp']:
        raise NotImplementedError('Varied parameter not implemented.')

##----------------------------------------------------------------------------##
## Dispatch Dictionaries                                                      ##
##----------------------------------------------------------------------------##
title_dispatch = {
    'I': 'Type I Error Comparison',
    'II': 'Type II Error Comparison'
}

x_lab_dispatch = {
    'n_samps': 'Number of Samples',
    'n_targs': 'Number of Targets',
    'disp':    r'Dispersion Parameter ($\gamma$)'
}

x_axis_vals_dispatch = {
    'n_samps': [20, 50, 100, 200, 500, 1000],
    'n_targs': [10, 20, 50, 100, 200, 500, 1000],
    'disp':    [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
}

update_dispatch = {
    # 'n_samps': update_n_samps,
    'n_targs': update_n_targs,
    'disp':    update_disp
}

error_dispatch = {
    'I': compute_type_i_err,
    'II': compute_type_ii_err,
}

settings_dispatch = {
    'I': check_type_i_settings,
    'II': check_type_ii_settings
}

##----------------------------------------------------------------------------##
## Computations                                                               ##
##----------------------------------------------------------------------------##
def error_comparison(
        d_config,
        m_config,
        iter_id: int,
        err_type: str,
        param:  str,
        level: float = 0.05,
        reps: int = 100
    ):
    """
    Compare test errors across varied parameter values.
    ARGS:
        d_config - str; path to data configuration .yaml
        m_config - str; path to model configuration .yaml
        err_type - str; error type (I or II)
        iter_id  - int; iteration ID (for labelling)
        param    - str; value along x-axis
        level    - float; test level (default: 0.05)
        reps     - int; number of replicates to draw for estimation (proportion)
    RAISES:  None
    RETURNS: None
    """
    ## check settings are appropriate
    settings_dispatch[err_type](d_config, m_config, param)

    ## get routine specifics by parameter being varied
    routine = f'type_{err_type}_error_by_{param}'
    x_axis_vals = x_axis_vals_dispatch[param]

    ## update seed for iteration
    d_config['seed'] = int(d_config['seed']) + iter_id

    ## get plot name and document routine
    out_dir = doc.document_routine(d_config, m_config, routine)

    ## initialize arrays of p-values
    oasis_p_vals   = np.zeros((reps, len(x_axis_vals)))
    oasis_asymp_ps = np.zeros((reps, len(x_axis_vals)))
    one_p_vals     = np.zeros((reps, len(x_axis_vals)))
    two_p_vals     = np.zeros((reps, len(x_axis_vals)))

    ## set error type function
    error_func = error_dispatch[err_type]
    update_param_func = update_dispatch[param]

    ## if there are no fixed effects, then we can also run OASIS (original) and get the asymptotic p-value
    if len(d_config['x']['types']) == 0:
        p_val_func = compute_p_vals_intercept_only
    else:
        p_val_func = compute_p_vals

    ## instantiate data
    data = data_types[d_config['outcome']](d_config)

    for r in tqdm(range(reps)):
        data.set_debug(iter = '', rep = str(r), routine = routine, seed = data.seed)
        for i in range(len(x_axis_vals)):
            update_param_func(data, x_axis_vals[i])
            model = model_types[m_config['outcome']](m_config, data)
            ## update debugging dictionary
            if m_config['debug'] == 'full':
                doc.debug_sample(model = model)
            doc.log_routine(model, routine)
            ## compute p-values
            oasis_p, asymp_p, one_p, two_p = p_val_func(data, model)
            oasis_p_vals[r, i]   = oasis_p
            oasis_asymp_ps[r, i] = asymp_p
            one_p_vals[r, i]     = one_p
            two_p_vals[r, i]     = two_p

    ## compute error
    oasis_err       = error_func(oasis_p_vals, level, reps)
    oasis_asymp_err = error_func(oasis_asymp_ps, level, reps)
    one_err         = error_func(one_p_vals, level, reps)
    two_err         = error_func(two_p_vals, level, reps)

    ## save results
    if len(d_config['x']['types']) == 0:
        jnp.savez(
            os.path.join(out_dir, routine, f'results_iter{iter_id}.npz'),
            x_axis_vals      = x_axis_vals,
            oasis_errs       = oasis_err,
            oasis_asymp_errs = oasis_asymp_err,
            one_sided_errs   = one_err,
            two_sided_errs   = two_err
        )
    else:
        jnp.savez(
            os.path.join(out_dir, routine, f'results_iter{iter_id}.npz'),
            x_axis_vals    = x_axis_vals,
            oasis_errs     = oasis_err,
            one_sided_errs = one_err,
            two_sided_errs = two_err
        )

##----------------------------------------------------------------------------##
## Plotting                                                                   ##
##----------------------------------------------------------------------------##
def process_results_intercept_only(dir_files, out_dir, routine):
    oasis_err   = []
    one_err     = []
    two_err     = []
    asymp_err   = []
    x_axis_vals = None
    for file in dir_files:
        iter_res = jnp.load(file)
        ## will be the same over all iterations, so we can just overwrite it
        x_axis_vals = iter_res['x_axis_vals']
        ## expand dims because we want to create a 2D array
        oasis_err.append(jnp.expand_dims(iter_res['oasis_errs'], 1))
        one_err.append(jnp.expand_dims(iter_res['one_sided_errs'], 1))
        two_err.append(jnp.expand_dims(iter_res['two_sided_errs'], 1))    
        asymp_err.append(jnp.expand_dims(iter_res['oasis_asymp_errs'], 1))
    oasis_errs = jnp.concatenate(oasis_err, axis = 1)
    one_errs   = jnp.concatenate(one_err, axis = 1)
    two_errs   = jnp.concatenate(two_err, axis = 1)
    asymp_errs = jnp.concatenate(asymp_err, axis = 1)
    ## save full results
    jnp.savez(
        os.path.join(out_dir, routine, f'results.npz'),
        x_axis_vals      = x_axis_vals,
        oasis_errs       = oasis_err,
        oasis_asymp_errs = asymp_err,
        one_sided_errs   = one_err,
        two_sided_errs   = two_err
    )
    ## take average over iterations
    oasis_errs = jnp.mean(oasis_errs, axis = 1)
    one_errs   = jnp.mean(one_errs, axis = 1)
    two_errs   = jnp.mean(two_errs, axis = 1)
    asymp_errs   = jnp.mean(asymp_errs, axis = 1)
    return x_axis_vals, oasis_errs, one_errs, two_errs, asymp_errs

def process_results(dir_files, out_dir, routine):
    oasis_err   = []
    one_err     = []
    two_err     = []
    x_axis_vals = None
    for file in dir_files:
        iter_res = jnp.load(file)
        ## will be the same over all iterations, so we can just overwrite it
        x_axis_vals = iter_res['x_axis_vals']
        ## expand dims because we want to create a 2D array
        oasis_err.append(jnp.expand_dims(iter_res['oasis_errs'], 1))
        one_err.append(jnp.expand_dims(iter_res['one_sided_errs'], 1))
        two_err.append(jnp.expand_dims(iter_res['two_sided_errs'], 1))    
    oasis_errs = jnp.concatenate(oasis_err, axis = 1)
    one_errs   = jnp.concatenate(one_err, axis = 1)
    two_errs   = jnp.concatenate(two_err, axis = 1)
    ## save full results
    jnp.savez(
        os.path.join(out_dir, routine, f'results.npz'),
        x_axis_vals      = x_axis_vals,
        oasis_errs       = oasis_err,
        one_sided_errs   = one_err,
        two_sided_errs   = two_err
    )
    ## take average over iterations
    oasis_errs = jnp.mean(oasis_errs, axis = 1)
    one_errs   = jnp.mean(one_errs, axis = 1)
    two_errs   = jnp.mean(two_errs, axis = 1)
    return x_axis_vals, oasis_errs, one_errs, two_errs, None

def plot_error_comparison(d_config, m_config, err_type, param, level, reps):
    """Save line plot of errors."""
    ## get routine specifics by parameter being varied
    routine = f'type_{err_type}_error_by_{param}'

    ## read all results from output directory 
    out_dir = doc.document_routine(d_config, m_config, routine)
    dir_files = glob.glob(os.path.join(out_dir, routine, 'results_iter*.npz'))
    if len(d_config['x']['types']) == 0:
        results_func = process_results_intercept_only
    else:
        results_func = process_results
    x_axis_vals, oasis_orig, one_sided, two_sided, asymp = results_func(
        dir_files = dir_files,
        out_dir   = out_dir, 
        routine   = routine
    )
    
    ## get necessary quantities/names
    plt_name = os.path.join(out_dir, routine, f'{routine}.pdf')
    plt.rcParams.update({'font.size': 8})
    _, axs = plt.subplots(
        1, 1,
        tight_layout = True, 
        figsize = (8, 8)
    )

    ## level alpha line
    axs.axline((0, level), slope = 0, color = 'black', linestyle = 'dotted', label = r'$\alpha$')

    ## plot results
    axs.plot(x_axis_vals, oasis_orig, color = 'green', label = 'OASIS')
    axs.plot(x_axis_vals, one_sided, color = 'blue', label = 'One-Sided')
    axs.plot(x_axis_vals, two_sided, color = 'red', label = 'Two-Sided')
    try:
        axs.plot(x_axis_vals, asymp, color = 'purple', label = 'OASIS (asymp.)')
    except:
        pass

    axs.legend()
    axs.set_xlabel(x_lab_dispatch[param])
    axs.set_ylabel(f'Estimated Type {err_type} Error',)
    axs.set_title(title_dispatch[err_type] + r' ($\alpha$ = ' + f'{level})')
    axs.semilogx(base = 10)
    plt.tight_layout()
    plt.savefig(plt_name)
    plt.close()

    if err_type == 'II':
        ## also plot as power if Type II error
        ## get necessary quantities/names
        plt_name = os.path.join(out_dir, routine, f'power_by_{param}.pdf')
        plt.rcParams.update({'font.size': 8})
        _, axs = plt.subplots(
            1, 1,
            tight_layout = True, 
            figsize = (8, 8)
        )

        ## compute power
        oasis_orig = [1 - x for x in oasis_orig]
        one_sided  = [1 - x for x in one_sided]
        two_sided  = [1 - x for x in two_sided]

        ## plot results
        axs.plot(x_axis_vals, oasis_orig, color = 'green', label = 'OASIS')
        axs.plot(x_axis_vals, one_sided, color = 'blue', label = 'One-Sided')
        axs.plot(x_axis_vals, two_sided, color = 'red', label = 'Two-Sided')
        try:
            asymp = [1 - x for x in asymp]
            axs.plot(x_axis_vals, asymp, color = 'purple', label = 'OASIS (asymp.)')
        except:
            pass

        axs.legend()
        axs.set_xlabel(x_lab_dispatch[param])
        axs.set_ylabel(f'Estimated Power',)
        axs.set_title(title_dispatch[err_type] + r' ($\alpha$ = ' + f'{level})')
        axs.semilogx(base = 10)
        plt.tight_layout()
        plt.savefig(plt_name)
        plt.close()
