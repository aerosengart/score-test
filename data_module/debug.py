
import os
import glob
import math
import re
import pickle
import jax

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import jax.numpy         as jnp
import numpy             as np

import matplotlib.cbook as cbook
import matplotlib.colors as colors

from stat_module import intermed, info, score, helpers

plt.rcParams['axes.unicode_minus']            = False


def print_dims(data):
    """Print dimensions of simulated data arrays related to the linear predictor.

    ARGUMENTS: data - instance of simul_data.Data
    RAISES:    None
    RETURNS:   None
    """
    x_dim = data.x_arr.shape
    z_dim = data.z_arr.shape
    fx_dim = data.fx_effs_arr.shape
    rd_dim = data.rd_effs_arr.shape

    print(f'{data.n_targs} targets (rows) and {data.n_samps} samples (columns)')
    print(f'X array: {x_dim}')
    print(f'Z array: {z_dim}')
    print(f'Fixed coefficients array: {fx_dim}')
    print(f'Random coefficients array: {rd_dim}')
    


def make_info_heatmap(arr, var, out_dir):
    plt.rcParams.update({'font.size': 8})
    fig, axs = plt.subplots(nrows = 1, ncols = 1)
    fig.tight_layout(rect = (0.02, 0.03, 1.0, 0.95))
    heatmap = axs.imshow(arr, interpolation = None, norm = colors.LogNorm())
    fig.colorbar(heatmap, ax = axs, shrink = 0.6)
    c_num = np.linalg.cond(arr)
    if var == 'info_mat':
        fig.suptitle(r'Information Matrix ($\kappa(\mathcal{I})$ = ' + f'{c_num})')
    else:
        fig.suptitle(r'Information Matrix Block Inverse')
    plt_name = os.path.join(out_dir, f'{var}.pdf')
    plt.savefig(plt_name)
    plt.close()   

def make_disp_heatmap(d_disp, m_disp, out_dir):
    ## get min and max
    combined_data = np.array([d_disp, m_disp])
    _min, _max = np.amin(combined_data) - 1, np.amax(combined_data) + 1

    plt.rcParams.update({'font.size': 8})
    fig, axs = plt.subplots(nrows = 1, ncols = 2, sharey = True, sharex = True)
    fig.tight_layout(rect = (0.02, 0.03, 1.0, 0.95))
    _ = axs[0].imshow(d_disp, interpolation = None, aspect='auto', vmin = _min, vmax = _max)
    heatmap = axs[1].imshow(m_disp, interpolation = None, aspect='auto', vmin = _min, vmax = _max)
    axs[0].set_title('data')
    axs[1].set_title('model')
    axs[0].set_xticks(np.arange(d_disp.shape[-1]))
    axs[1].set_xticks(np.arange(m_disp.shape[-1]))
    axs[0].tick_params(labelbottom = False) 
    axs[1].tick_params(labelbottom = False) 

    fig.colorbar(heatmap, ax = axs, shrink = 0.6)
    fig.suptitle(r'Dispersion Parameter ($\gamma$)')
    fig.supylabel('targets')
    plt_name = os.path.join(out_dir, f'dispersion.pdf')
    plt.savefig(plt_name)
    plt.close()   


def make_intermed_heatmap(arr, title, var, out_dir):
    plt.rcParams.update({'font.size': 8})
    fig, axs = plt.subplots(nrows = 1, ncols = 1)
    fig.tight_layout(rect = [0.02, 0.03, 1, 0.95])
    heatmap = axs.imshow(arr, interpolation = None, aspect='auto', vmin = np.min(arr), vmax = np.max(arr))
    axs.set_xticks(np.arange(arr.shape[-1]))
    axs.tick_params(labelbottom = False) 
    fig.colorbar(heatmap, ax = axs, shrink = 0.6)
    fig.suptitle(title)
    plt_name = os.path.join(out_dir, f'{var}.pdf')
    plt.savefig(plt_name)
    plt.close()   


def make_coefs_heatmap(d_fx_arr, d_rd_arr, m_fx_arr, m_rd_arr, out_dir):
    ## get min and max
    combined_data = np.array([d_fx_arr, d_rd_arr, m_fx_arr, m_rd_arr])
    _min, _max = np.amin(combined_data), np.amax(combined_data)

    plt.rcParams.update({'font.size': 8})
    fig, axs = plt.subplots(nrows = 2, ncols = 2, sharey = 'row', sharex = 'col')
    fig.tight_layout(rect = (0.02, 0.03, 1.0, 0.95))
    _ = axs[0, 0].imshow(d_fx_arr, interpolation = None, aspect='auto', vmin = _min, vmax = _max)
    _ = axs[0, 1].imshow(m_fx_arr, interpolation = None, aspect='auto', vmin = _min, vmax = _max)
    _ = axs[1, 0].imshow(d_rd_arr, interpolation = None, aspect='auto', vmin = _min, vmax = _max)
    heatmap = axs[1, 1].imshow(m_rd_arr, interpolation = None, aspect='auto', vmin = _min, vmax = _max)
    
    axs[0, 0].set_title(r'data fixed coefs ($\alpha$)')
    axs[0, 0].set_ylabel('targets')
    axs[0, 1].set_title(r'model fixed coefs ($\alpha$)')
    axs[0, 1].set_ylabel('targets')
    axs[1, 0].set_title(r'data random coefs ($\beta$)')
    axs[1, 0].set_ylabel('targets')
    axs[1, 1].set_title(r'model random coefs ($\beta$)')
    axs[1, 1].set_ylabel('targets')

    axs[0, 0].set_xticks(np.arange(d_fx_arr.shape[-1]))
    axs[0, 1].set_xticks(np.arange(m_fx_arr.shape[-1]))
    axs[1, 0].set_xticks(np.arange(d_rd_arr.shape[-1]))
    axs[1, 1].set_xticks(np.arange(m_rd_arr.shape[-1]))

    axs[0, 0].tick_params(labelbottom = False) 
    axs[0, 1].tick_params(labelbottom = False) 
    axs[1, 0].tick_params(labelbottom = False) 
    axs[1, 1].tick_params(labelbottom = False) 

    fig.colorbar(heatmap, ax = axs, shrink = 0.6)
    fig.suptitle('Coefficients')
    plt_name = os.path.join(out_dir, f'coefs.pdf')
    plt.savefig(plt_name)
    plt.close()   



def make_eta_mu_heatmap(d_eta_arr, d_mu_arr, m_eta_arr, m_mu_arr, out_dir):
    ## get min and max
    combined_data = np.array([d_eta_arr, m_eta_arr])
    eta_min, eta_max = np.amin(combined_data), np.amax(combined_data)
    combined_data = np.array([d_mu_arr, m_mu_arr])
    mu_min, mu_max = np.amin(combined_data), np.amax(combined_data)

    plt.rcParams.update({'font.size': 8})
    fig, axs = plt.subplots(nrows = 2, ncols = 2, sharey = 'row', sharex = 'col')
    fig.tight_layout(rect = (0.02, 0.03, 1.0, 0.95))
    _ = axs[0, 0].imshow(d_eta_arr, interpolation = None, aspect='auto', vmin = eta_min, vmax = eta_max)
    heatmap1 = axs[0, 1].imshow(m_eta_arr, interpolation = None, aspect='auto', vmin = eta_min, vmax = eta_max)
    _ = axs[1, 0].imshow(d_mu_arr, interpolation = None, aspect='auto', vmin = mu_min, vmax = mu_max)
    heatmap2 = axs[1, 1].imshow(m_mu_arr, interpolation = None, aspect='auto', vmin = mu_min, vmax = mu_max)
    
    axs[0, 0].set_title(r'data linear predictors ($\eta$)')
    axs[0, 1].set_title(r'model linear predictors ($\eta$)')
    axs[1, 0].set_title(r'data means ($\mu$)')
    axs[1, 1].set_title(r'model means ($\mu$)')
    axs[0, 0].set_xticks(np.arange(d_eta_arr.shape[-1]))
    axs[0, 1].set_xticks(np.arange(m_eta_arr.shape[-1]))
    axs[1, 0].set_xticks(np.arange(d_mu_arr.shape[-1]))
    axs[1, 1].set_xticks(np.arange(m_mu_arr.shape[-1]))

    axs[0, 0].tick_params(labelbottom = False) 
    axs[0, 1].tick_params(labelbottom = False) 
    axs[1, 0].tick_params(labelbottom = False) 
    axs[1, 1].tick_params(labelbottom = False) 

    fig.colorbar(heatmap1, ax = axs[0, :], shrink = 0.6)
    fig.colorbar(heatmap2, ax = axs[1, :], shrink = 0.6)
    fig.suptitle('Linear Predictors and Means')
    fig.supylabel('targets')
    fig.supxlabel('samples')
    plt_name = os.path.join(out_dir, f'eta_mu.pdf')
    plt.savefig(plt_name)
    plt.close()   


def make_counts_heatmap(y_arr, x_arr, z_arr, out_dir):
    combined_data = np.array([x_arr, z_arr])
    _min, _max = np.amin(combined_data), np.amax(combined_data)

    plt.rcParams.update({'font.size': 8})
    fig, axs = plt.subplots(ncols = 3, figsize = (10, 6))
    fig.tight_layout(rect=[0.02, 0.03, 1, 0.95])
    heatmap = axs[0].imshow(y_arr, interpolation = None, aspect = 'auto', vmin = 0, vmax = np.max(y_arr))
    _ = axs[1].imshow(x_arr, interpolation = None, aspect = 'auto', vmin = _min, vmax = _max) 
    heatmap2 = axs[2].imshow(z_arr, interpolation = None, aspect = 'auto', vmin = _min, vmax = _max)
    axs[0].set_title(r'counts ($y$)')
    axs[0].set_xlabel('samples')
    axs[0].set_ylabel('targets')
    axs[1].set_title(r'fixed effs covars ($x$)')
    axs[1].set_xlabel('p')
    axs[1].set_ylabel('samples')
    axs[2].set_title(r'random effs covars ($z$)')
    axs[2].set_xlabel('q')
    axs[2].set_ylabel('samples')
    axs[0].set_xticks(np.arange(y_arr.shape[-1]))
    axs[1].set_xticks(np.arange(x_arr.shape[-1]))
    axs[2].set_xticks(np.arange(z_arr.shape[-1]))
    axs[0].tick_params(labelbottom = False) 
    axs[1].tick_params(labelbottom = False) 
    axs[2].tick_params(labelbottom = False) 

    fig.colorbar(heatmap, ax = axs[0], shrink = 0.6)
    #fig.colorbar(heatmap1, ax = axs[1], shrink = 0.6)
    fig.colorbar(heatmap2, ax = axs[1:], shrink = 0.6)
    fig.suptitle('Raw Data')
    plt_name = os.path.join(out_dir, f'sampled_data.pdf')
    plt.savefig(plt_name)
    plt.close()


def visualize_data(model_pickle):
    ## data_pickle is path to data -- get outer directory
    split_path = model_pickle.rsplit('/', 1)
    outer_dir = split_path[0]
    name = split_path[1].split('.')[0].split('_')[0]
    ## make out_dir in path where pickle is
    out_dir = os.path.join(outer_dir, name)
    ## check if save path exists
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok = True)
    ## load data
    file = open(model_pickle, 'rb')
    model = pickle.load(file)
    file.close()

    ## plot raw data
    data = model.data
    make_counts_heatmap(data.construct_y(), data.x_arr, data.z_arr, out_dir)

    ## linear predictor/mean
    make_eta_mu_heatmap(data.compute_eta(), data.compute_mu(), model.compute_eta(), model.compute_mu(), out_dir)

    ## plot coefficients
    make_coefs_heatmap(model.data.fx_effs_arr, model.data.rd_effs_arr, model.fx_effs_arr, model.rd_effs_arr, out_dir)

    ## plot dispersion parameters
    make_disp_heatmap(data.disp_arr, model.disp_arr, out_dir)

    ## plot intermediate quantities
    make_intermed_heatmap(model.compute_resid(), r'Residuals ($y - \mu$)', var = 'resids', out_dir = out_dir)
    make_intermed_heatmap(intermed.compute_delta(model, inv = False), r'Delta Values ($\delta$)', var = 'delta', out_dir = out_dir)
    make_intermed_heatmap(intermed.compute_omega(model), r'Omega Values ($\omega$)', var = 'omega', out_dir = out_dir)
    make_intermed_heatmap(intermed.compute_xi(model), r'Xi Values ($\xi$)', var = 'xi', out_dir = out_dir)
    make_intermed_heatmap(intermed.compute_epsilon(model), r'Epsilon Values ($\epsilon$)', var = 'epsilon', out_dir = out_dir)
    make_intermed_heatmap(intermed.compute_delta_inv_omega(model), r'Delta Inverse Omega Values ($\delta^{-1}\omega$)', var = 'delta_inv_omega', out_dir = out_dir)
    make_intermed_heatmap(intermed.compute_delta_inv_omega(model) * model.compute_resid(), r'Residual Delta Inverse Omega Values ($(y - \mu) \delta^{-1}\omega$)', var = 'resid_delta_inv_omega', out_dir = out_dir)

    ## compute score
    score_val, scale_val = score.compute_score_var_comp(
        model, 
        scale = model.scale
    )
    ## compute information
    info_fx, info_fx_var, info_var = info.compute_info(
        model, 
        scale_val = scale_val
    )

    ## schur complement
    schur_inv = helpers.lower_right_inv(
        upp_left  = info_fx,
        middle    = info_fx_var,
        low_right = info_var
    )

    ## compute stat
    if model.data.q == 1:
        stat = score_val**2 * schur_inv
    else:
        stat = np.matmul(
            np.matmul(np.transpose(score_val), schur_inv), score_val
        )

    ## plot information
    info_fx_list = [info_fx[i, :, :] for i in range(data.n_targs)]
    info_fx_full = jax.scipy.linalg.block_diag(*info_fx_list)
    info_fx_var_list = [info_fx_var[i, :, :] for i in range(data.n_targs)]
    info_fx_var_full = np.concatenate(info_fx_var_list, axis = 0)
    info_mat = np.block([[info_fx_full, info_fx_var_full],[info_fx_var_full.T, info_var]])
    make_info_heatmap(info_mat, 'info_mat', out_dir)


    make_intermed_heatmap(np.squeeze(info_fx, axis = 1), 'Information Matrix - Fixed Effects, Fixed Effects', 'info_fx', out_dir)
    make_intermed_heatmap(np.squeeze(info_fx_var, axis = 1), 'Information Matrix - Fixed Effects, Variance Component(s)', 'info_fx_var', out_dir)


    ## compute score
    score_val2, scale_val2 = score.compute_score_var_comp(
        model, 
        scale = False
    )
    ## compute information
    info_fx2, info_fx_var2, info_var2 = info.compute_info(
        model, 
        scale_val = scale_val2
    )

    ## schur complement
    schur_inv2 = helpers.lower_right_inv(
        upp_left  = info_fx2,
        middle    = info_fx_var2,
        low_right = info_var2
    )

    ## compute stat
    if model.data.q == 1:
        stat2 = score_val2**2 * schur_inv2
    else:
        stat2 = np.matmul(
            np.matmul(np.transpose(score_val2), schur_inv2), score_val2
        )

    ## compute target log-liks
    scores = np.zeros((model.data.n_targs, model.data.q))
    term1s = np.zeros((model.data.n_targs, model.data.q))
    term2s = np.zeros((model.data.n_targs, model.data.q))

    ## initial computations
    resid_vals     = model.compute_resid()
    xi_vals        = intermed.compute_xi(model)
    delta_inv_omega_vals = intermed.compute_delta_inv_omega(model)
    for idx in range(model.data.q):
        ## (n_samps, n_samps) --- same for all targets
        a_j_arr = intermed.compute_a_j(model, j = idx)

        ## vectorize wrapper --- compute target-by-target
        def wrap_func2(resid_targ, delta_inv_omega_targ, xi_targ):
            rdo_targ = resid_targ * delta_inv_omega_targ
            term1 = rdo_targ.T @ a_j_arr @ rdo_targ
            return term1
        
            ## vectorize wrapper --- compute target-by-target
        def wrap_func3(resid_targ, delta_inv_omega_targ, xi_targ):
            term2 = np.diag(xi_targ) @ a_j_arr
            return np.trace(term2)
    
        ## compute values
        vmap_func2 = jax.vmap(wrap_func2, in_axes = [0, 0, 0])
        vmap_func3 = jax.vmap(wrap_func3, in_axes = [0, 0, 0])
        summands2 = vmap_func2(resid_vals, delta_inv_omega_vals, xi_vals)
        summands3 = vmap_func3(resid_vals, delta_inv_omega_vals, xi_vals)
        term1s[:, idx] = summands2
        term2s[:, idx] = summands3
        scores[:, idx] = (summands2 - summands3) / 2

    make_intermed_heatmap(scores, 'Score Contributions by Target', var = 'target_score', out_dir = out_dir)
    make_intermed_heatmap(term1s, 'Score Term 1 by Target', var = 'target_score_term1', out_dir = out_dir)
    make_intermed_heatmap(term2s, 'Score Term 2 by Target', var = 'target_score_term2', out_dir = out_dir)

    mean_val = np.mean(info_fx)
    upp_block = info_fx / mean_val
    mid_scale = info_fx_var / np.sqrt(mean_val)
    dets = np.linalg.det(upp_block)
    def wrap_pinv(mat):
        return np.linalg.pinv(mat)
    pinv_func = jax.vmap(wrap_pinv, in_axes = [0])
    ## (rel_targs, p, p)
    upp_block_invs = pinv_func(upp_block)
    info_fx_inv_list = [upp_block_invs[i, :, :] for i in range(data.n_targs)]
    info_fx_inv_full = jax.scipy.linalg.block_diag(*info_fx_inv_list)
    make_intermed_heatmap(info_fx_inv_full, 'Information Matrix Block Inverse', 'info_fx_inv', out_dir)

    ## D - C A^{-1} B = M/A 
    def wrap_func(c_i, a_i):
        ## c_i is (p, q)
        ## a_i is (p, p)
        return np.matmul(np.matmul(c_i.T, a_i), c_i)
    vmap_func = jax.vmap(wrap_func, in_axes = [0, 0])
    vals = vmap_func(mid_scale, upp_block_invs)
    make_intermed_heatmap(vals, 'Schur Complement Values', 'schur_comp_vals', out_dir)

    with open(os.path.join(out_dir, 'other_vals.txt'), 'w') as file:
        file.write(f'Score Vector (unscaled): {round(score_val2, 5)}\n')
        file.write(f'Score Vector (scaled): {round(score_val, 5)}\n')
        file.write(f'Scale Value: {round(scale_val, 5)}\n')
        file.write(f'Efficient Information Inverse (unscaled): {round(schur_inv2, 5)} \n')
        file.write(f'Efficient Information Inverse (scaled): {round(schur_inv, 5)} \n')
        file.write(f'Two-Sided Statistic (unscaled): {round(stat2, 5)} \n')
        file.write(f'Two-Sided Statistic (scaled): {round(stat, 5)} \n')


