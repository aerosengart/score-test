
import os
import glob
import math
import re
import pickle
import jax

import matplotlib.pyplot as plt
import jax.numpy         as jnp
import numpy             as np

import seaborn as sns


from data_module.dispatch import data_types
from model_module.dispatch import model_types


plt.rcParams['axes.unicode_minus'] = False

def plot_mu_arr(out_dir):
    """Plot mu array for case when beta ~ N(2, 0)."""
    d_config = {
        'outcome': 'neg_bin',
        'seed': 7647766,
        'n_samps': 20,
        'n_targs': 25,
        'fx_effs': {
            'mu': 0.5,
            'sigma': 0.0
        },
        'rd_effs': {
            'mu': 2.0,
            'sigma': [0.0]
        },
        'x': {
            'center': False,
            'types': []
        },
        'z': {
            'center': True,
            'types': ['time_reg']
        },
        'disp': {
            'mu': 2.0,
            'sigma': 0.0
        }
    }

    m_config = {
        'outcome': 'neg_bin',
        'use_true': False,
        'scale': True,
        'global_disp': False,
        'debug': 'none',
        'out_dir': out_dir,
    }

    data = data_types[d_config['outcome']](d_config)
    data.resample()
    model = model_types[m_config['outcome']](m_config, data)
    d_mu_arr = data.compute_mu()
    ## reorder columns according to z
    col_idx = np.argsort(data.z_arr, axis = 0)
    d_mu_arr_sort = d_mu_arr[:, col_idx]
    m_mu_arr_sort = model.compute_mu()[:, col_idx]
    combined_data = np.array([d_mu_arr_sort, m_mu_arr_sort])
    _min, _max = np.amin(combined_data) - 1, np.amax(combined_data) + 1
    ## plot
    plt.rcParams.update({'font.size': 8})
    fig, axs = plt.subplots(nrows = 1, ncols = 2, sharex = True, sharey = True)
    fig.tight_layout(rect = [0.02, 0.03, 1, 0.95])
    _ = axs[0].imshow(d_mu_arr_sort, interpolation = None, vmin = _min, vmax = _max)
    heatmap = axs[1].imshow(m_mu_arr_sort, interpolation = None, vmin = _min, vmax = _max)
    axs[0].set_title(r'$\mu = \exp\left(\alpha_{0,i} + \beta_{1,i} z_{j,1}\right)$')
    axs[1].set_title(r'$\hat{\mu} = \exp\left(\hat{\nu}_j + \hat{\alpha}_{0,i} \right)$')
    fig.supxlabel('Samples')
    fig.supylabel('Targets')
    fig.colorbar(heatmap, ax = axs, shrink = 0.6)
    plt.savefig(os.path.join(out_dir, 'mu_arr_beta2_tau0.pdf'))  
    plt.close()   

    