"""Single iteration routines for assessing empirical distributions of various quantities."""

import os
import glob

import matplotlib.pyplot as plt
import jax.numpy         as jnp
import numpy             as np

from tqdm        import tqdm

from routine_module  import doc
from stat_module import score, helpers

from data_module.dispatch  import data_types
from model_module.dispatch import model_types

## some sort of issue with matplotlib, latex, and the unicode minus sign???
#plt.rcParams['text.usetex'] = True
plt.rcParams['axes.unicode_minus'] = False

##----------------------------------------------------------------------------##
## Computations                                                               ##
##----------------------------------------------------------------------------##

def score_var_comp_emp_dist(
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
    ## update seed for iteration
    d_config['seed'] = int(d_config['seed']) + iter_id

    ## instantiate data
    data = data_types[d_config['outcome']](d_config)
    data.resample()
    ## initialize
    scores_arr = jnp.zeros((data.q, reps)) 
    routine = 'score_var_comp_emp_dist'
    ## get plot name and document routine
    out_dir = doc.routine_dir(d_config, m_config, routine)

    for i in tqdm(range(reps)):
        ## draw observations and build model
        data.set_debug(iter = '', rep = str(i), routine = routine, seed = data.seed)
        data.resample()
        model = model_types[m_config['outcome']](m_config, data)
        ## update debugging dictionary
        if m_config['debug'] == 'full':
            doc.save_rep(model, d_config, m_config, sub_dir = None)
        doc.log_routine(model, d_config, m_config, routine)
        ## compute score
        gradient, _ = score.compute_score_var_comp(model, scale = False)
        scores_arr = scores_arr.at[:, i].set(gradient)

    ## save full results
    jnp.savez(
        os.path.join(out_dir, f'results_iter{iter_id}.npz'),
        scores = scores_arr
    )
    return out_dir


def score_var_comp_by_disp_emp_dist(
        d_config,
        m_config,
        iter_id: int,
        reps:    int = 100
    ):
    """Plot score function for each component of :math:`\tau^2` to check approximate Gaussianity and zero expectation under the null hypothesis over a grid of jitter values for the negative binomial dispersion parameter.

    ARGS:
        d_config - str; path to data configuration .yaml
        m_config - str; path to model configuration .yaml
        iter_id  - int; iteration ID (for labelling)
        reps     - int; number of replicates 
    RAISES:  None
    RETURNS: None
    """
    ## instantiate data
    data = data_types[d_config['outcome']](d_config)
    ## initialize
    disp_vals = [-0.5, -0.25, -0.1, -0.05, 0.0, \
                  0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    scores_arr = jnp.zeros((data.q, len(disp_vals), reps)) 
    routine = 'score_var_comp_by_disp_emp_dist'
    ## get plot name and document routine
    out_dir = doc.routine_dir(d_config, m_config, routine)
    ## copy model config
    m_config_alt = m_config
    for i in tqdm(range(reps)):
        data.set_debug(iter = '', rep = str(i), routine = routine, seed = data.seed)
        data.resample()
        for j in range(len(disp_vals)):
            ## draw observations and build model
            m_config_alt['global_disp'] = data.disp_arr + disp_vals[j]
            model = model_types[m_config_alt['outcome']](m_config_alt, data)
            ## update debugging dictionary
            if m_config['debug'] == 'full':
                doc.debug_sample(model = model)
            ## compute score
            gradient, _ = score.compute_score_var_comp(model, scale = False)
            scores_arr = scores_arr.at[:, i].set(gradient)

    ## save full results
    jnp.savez(
        os.path.join(out_dir, routine, f'results_iter{iter_id}.npz'),
        scores = scores_arr
    )
    return out_dir


##----------------------------------------------------------------------------##
## Plotting                                                                   ##
##----------------------------------------------------------------------------##

def score_hist(d_config, m_config, reps):
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
    dim = len(d_config['rd_effs']['sigma'])
    plt_name = os.path.join(out_dir, routine, 'score_var_comp.pdf')

    if dim == 1:
        ## variance component is scalar-valued
        _, axs = plt.subplots(
            1, 1, 
            tight_layout = True,
            figsize = (10, 6)
        )
        axs.hist(scores_arr.tolist(), bins = 100, density = True)    
        axs.set_title(r'analytical score component of $\tau^2$ under $H_0$')
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
            axs[i].set_title(r'analytical score component of $\tau^2_{{{i}}}$ under $H_0$')
            axs[i].set_xlabel('score')
            axs[i].set_ylabel('density')
    plt.savefig(plt_name)
    plt.close()
