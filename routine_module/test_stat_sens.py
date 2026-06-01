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

def test_stat_fx_eff_sensitivity(
        d_config,
        m_config,
        iter_id:     int,
        reps:        int = 100,
        disp_known:  bool = False,
    ):
    """
    Compute score test statistic for jittered values of the fixed effects coefficients. Optionally, hold negative binomial/quasi-Poisson dispersion parameter fixed at its true value.

    ARGS:
        d_config    - data configuration dictionary
        m_config    - model configuration dictionary
        iter_id     - positive int: iteration number for output name
        reps        - int; number of replicates to run per iteration
        disp_known  - bool; fix dispersion at true value
    RAISES:  None
    RETURNS: None
    """
    if d_config['outcome'] == 'splash':
        raise Exception('Cannot jitter real data')
    
    ## don't estimate because we are setting the fixed effects directly
    new_config = m_config
    new_config['use_true'] = True

    ## routine name
    if disp_known:
        routine = 'test_stat_fx_effs_sens_disp_known'
    else:
        routine = 'test_stat_fx_effs_sens'

    ## document routine
    out_dir = doc.document_routine(d_config, m_config, routine)

    ## instantiate data
    data = data_types[d_config['outcome']](d_config)

    ## initialize
    x_axis_ticks = [-0.1, -0.05, -0.025, -0.01, 0.0, \
                    0.01, 0.025, 0.05, 0.1]
    one_sided = jnp.zeros((reps, len(x_axis_ticks)))
    two_sided = jnp.zeros((reps, len(x_axis_ticks)))


    for r in tqdm(range(reps)):
        ## draw data and build model
        data.resample()
        data.set_debug(iter = str(iter_id), rep = str(r), routine = routine)
        model = model_types[m_config['outcome']](m_config, data)
        ## set dispersion parameter
        if disp_known:
            if m_config['outcome'] == 'neg_bin':
                model.set_disp_param(data.gamma)
            else:
                model.set_disp_param(data.phi)

        ## update debugging dictionary
        if m_config['debug'] == 'full':
            doc.debug_sample(model = model)

        for k in range(len(x_axis_ticks)):
            ## jitter the fixed effects
            fx_effs_arr = data.fx_effs_arr + x_axis_ticks[k]
            ## overwrite model MLE with jittered array
            model.set_fx_effs_arr(fx_effs_arr)
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



# def test_stat_gamma_sensitivity(
#         config,
#         fx_effs_known: bool = False,
#         reps:          int  = 100,
#         iters:         int  = 1
#     ):
#     """Plot a grid of QQ-plots likelihood ratio and score test statistics evaluated with jittered values of the negative binomial scale/dispersion parameter, gamma. 

#     ARGS:
#         config - a .yaml file specifying the settings for sampling data
#         reps   - integer; number of replicates to run for histogram
#     RETURNS:
#         None
#     RAISES:
#         None
#     """
#     if config['data']['outcome'] == 'real':
#         raise Exception('Cannot jitter real data')
#     ## don't estimate because we are setting the fixed effects directly
#     new_config = config
#     new_config['model']['use_true'] = True

#     ## routine name
#     if fx_effs_known:
#         routine = 'test_stat_gamma_sens_fx_effs_known'
#     else:
#         routine = 'test_stat_gamma_sens'

#     ## initialize
#     x_axis_ticks = [-0.5, -0.25, -0.1, -0.05, 0.0, \
#                     0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
#     data = neg_bin_data[config['data']['spec']](new_config)
    
#     ## get plot name and document routine
#     out_dir = doc.document_routine(
#         data = data,
#         routine = routine,
#         print_dims = True
#     )

#     ## construct plot name
#     plt_name = os.path.join(out_dir, routine, routine) + '.pdf'
#     lrt_all_iters   = jnp.zeros((iters, reps, len(x_axis_ticks)))
#     score_all_iters = jnp.zeros((iters, reps, len(x_axis_ticks)))

#     ## construct saved results name
#     res_name = os.path.join(out_dir, routine, 'results.npz')

#     if not os.path.isfile(res_name):
#         for i in range(iters):
#             print(f'==============ITERATION {i}==============\n')
#             lrt_stats   = jnp.zeros((reps, len(x_axis_ticks)))
#             score_stats = jnp.zeros((reps, len(x_axis_ticks)))
#             for j in tqdm(range(reps)):
#                 ## draw data and build model
#                 data.sample()
#                 data.set_debug(iter = str(i), rep = str(j), routine = routine)
#                 model = model_types[new_config['model']['outcome']](data)
#                 if fx_effs_known:
#                     fx_effs_arr = data.fx_effs_arr
#                 for k in range(len(x_axis_ticks)):
#                     ## jitter gamma
#                     gamma = data.gamma + x_axis_ticks[k]
#                     ## compute test statistics
#                     lrt_stat = lrt.compute_test_statistic(
#                         model         = model,
#                         fx_effs_arr_n = fx_effs_arr, ## needs to be adjusted for 2d case to not include fx slope
#                         fx_effs_arr_a = fx_effs_arr,
#                         gamma         = gamma
#                     )
#                     ## overwrite model MLE with jittered array
#                     model.gamma_mle = gamma
#                     if fx_effs_known:
#                         model.fx_effs_mle_arr = fx_effs_arr
#                     score_stat = score.compute_test_stat(model, one_sided = True)
#                     ## record values
#                     lrt_stats   = lrt_stats.at[j, k].set(lrt_stat)
#                     score_stats = score_stats.at[j, k].set(score_stat)
#             lrt_all_iters = lrt_all_iters.at[i, :, :].set(lrt_stats)
#             score_all_iters = score_all_iters.at[i, :, :].set(score_stats)
#             ## save each iteration
#             jnp.savez(
#                 os.path.join(out_dir, routine, f'results_iter{i}.npz'),
#                 lrt_stats = lrt_stats,
#                 score_stats = score_stats
#             )
#         ## save full results array and delete iterations
#         jnp.savez(
#             os.path.join(out_dir, routine, f'results.npz'),
#             lrt_stats = lrt_all_iters,
#             score_stats = score_all_iters
#         )
#         for filename in glob.glob(os.path.join(
#             out_dir, routine, 'results_iter*'
#         )):
#             os.remove(filename) 
#     else:
#         results = jnp.load(res_name)
#         lrt_stats = results['lrt_stats']
#         score_stats = results['score_stats']
#     ## plot
#     plot.lrt_score_jitter_grid(
#         lrt_stats = lrt_stats, 
#         score_stats = score_stats, 
#         q = data.q,
#         k = data.k, 
#         x_axis_ticks = x_axis_ticks, 
#         plt_name = plt_name
#     )


##----------------------------------------------------------------------------##
## Plotting                                                                   ##
##----------------------------------------------------------------------------##

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