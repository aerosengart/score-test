import os
import shutil
import yaml

import jax.numpy     as jnp
import pickle


def log_configs(r_path, out_dir):
    save_dir = os.path.join(out_dir, 'configs')

    ## check if save path exists
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok = True)

    with open(r_path, 'r') as file:
        r_config = yaml.safe_load(file)
    
    ## copy configuration files
    shutil.copy(r_path, os.path.join(save_dir, 'r_config.yaml')) 
    shutil.copy(r_config['d_config'], os.path.join(save_dir, 'd_config.yaml'))
    shutil.copy(r_config['m_config'], os.path.join(save_dir, 'm_config.yaml'))


def output_directory_path(d_config, m_config):
    """Construct appropriate sub-directory for output.

    ARGS:
        d_config - data configuration dictionary
        m_config - model configuration dictionary
    RAISES:
        None
    RETURNS:
        out_dir - output sub-directory path
    """
    n = str(d_config['n_samps'])
    k = str(d_config['n_targs'])
    if 'fx_slopes' in d_config.keys():
        p = str(1 + d_config['fx_slopes'])
        q = str(d_config['rd_slopes'])
    else:
        p = str(1 + len(d_config['x']['types']))
        q = str(len(d_config['z']['types']))

    ## data outcome distribution
    data_dist = 'd{}_'.format(d_config['outcome'])

    ## model outcome distribution
    model_dist = 'm' + m_config['outcome'] + '_'

    ## fixed part
    fx_part = f'fx_p{p}_'
    ## random part
    rd_part = f'rd_q{q}_'
    ## dimension part
    dim_part = f'samps{n}_targs{k}'
    ## concatenate
    sub_dir = data_dist + model_dist + fx_part + rd_part + dim_part

    sub_dir = os.path.join(m_config['out_dir'], sub_dir)
    return sub_dir


def format_attrbs(val):
    attrbs = []
    if isinstance(val, list):
        for y in val:
            if isinstance(y, list):
                attrbs.append('[' + ', '.join([str(round(x, 3)) for x in y]) + ']')
            else:
                attrbs.append(str(round(y, 3)))
    else:
        attrbs.append(str(round(val, 3)))
    return ', '.join(attrbs)


def log_routine(model, d_config, m_config, routine):
    """Document the routine ran in a README file.
    
    ARGS:
        data       - an instance of Data class
        model      - an instance of Model class
        routine    - str; routine name
    RAISES:
        None
    RETURNS:
        output directory
    """
    ## get output directory path
    out_dir = output_directory_path(d_config, m_config)
    routine_dir = os.path.join(out_dir, routine)
    ## add sub-directories for iteration
    save_dir = os.path.join(
        routine_dir,
        f'iter{model.data.debug['iter']}'
    )

    ## check if save path exists
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok = True)

    with open(os.path.join(save_dir, 'log.txt'), "a") as file:
        mu_arr = model.data.compute_mu()
        file.write(f'{model.data.n_targs} Targets, {model.data.n_samps} Samples, Seed = {model.data.seed} \n')
        file.write(f'Data Targets 0 and 1: \n')
        file.write(f'    Mean {mu_arr.shape}: {format_attrbs(mu_arr[0:2, 0:2].tolist())}\n')
        file.write(f'    Fixed Effects {model.data.fx_effs_arr.shape}: {format_attrbs(model.data.fx_effs_arr[0:2, :].tolist())}\n')
        file.write(f'    Random Effects {model.data.rd_effs_arr.shape}: {format_attrbs(model.data.rd_effs_arr[0:2, :].tolist())}\n')
        file.write(f'    Dispersion {model.data.disp_arr.shape}: {format_attrbs(model.data.disp_arr[0:2, :].tolist())}\n')
        file.write(f'    Random Effects Mean and SD: {format_attrbs(model.data.rd_effs_mu), format_attrbs(model.data.rd_effs_sigma)}\n')
        file.write(f'Model Targets 0 and 1: \n')
        file.write(f'    Mean {model.mu_arr.shape}: {format_attrbs(model.mu_arr[0:2, 0:2].tolist())}\n')
        file.write(f'    Fixed Effects {model.fx_effs_arr.shape}: {format_attrbs(model.fx_effs_arr[0:2, :].tolist())}\n')
        file.write(f'    Random Effects {model.rd_effs_arr.shape}: {format_attrbs(model.rd_effs_arr[0:2, :].tolist())}\n')
        file.write(f'    Dispersion {model.disp_arr.shape}: {format_attrbs(model.disp_arr[0:2, :].tolist())}\n')
        file.write(f'    Variance Component: {format_attrbs(model.var_comp[0:2].tolist())}\n')
        file.write('================================================================================\n')



def document_routine(d_config, m_config, routine):
    """Document the routine ran in a README file.
    
    ARGS:
        data       - an instance of Data class
        model      - an instance of Model class
        routine    - str; routine name
    RAISES:
        None
    RETURNS:
        output directory
    """
    ## get output directory path
    r_dir = routine_dir(d_config, m_config, routine)

    ## write README file
    readme_name = os.path.join(r_dir, 'config.txt')
    with open(readme_name, 'w') as f:
        f.write('========================================================== \n')
        f.write('Data Configuration \n')
        f.write('========================================================== \n')
        for k, v in d_config.items():
            if isinstance(v, dict):
                f.write(str(k) + ': \n')
                for k1, v1 in v.items():
                    if isinstance(v1, dict):
                        f.write('    ' + str(k1) + ': \n')
                        for k2, v2 in v1.items():
                            f.write('        ' + str(k2) + ': ' + str(v2) + '\n')
                    else:
                        f.write('    ' + str(k1) + ': ' + str(v1) + '\n')
            else:
                f.write(str(k) + ': '  + str(v) + '\n')

        f.write('\n \n')
        f.write('========================================================== \n')
        f.write('Model Configuration \n')
        f.write('========================================================== \n')
        for k, v in m_config.items():
            if isinstance(v, dict):
                f.write(str(k) + ': \n')
                for k1, v1 in v.items():
                    if isinstance(v1, dict):
                        f.write('    ' + str(k1) + ': \n')
                        for k2, v2 in v1.items():
                            f.write('        ' + str(k2) + ': ' + str(v2) + '\n')
                    else:
                        f.write('    ' + str(k1) + ': ' + str(v1) + '\n')
            else:
                f.write(str(k) + ': '  + str(v) + '\n')
    return r_dir

def routine_dir(d_config, m_config, routine):
    ## get output directory path
    out_dir = output_directory_path(d_config, m_config)
    routine_dir = os.path.join(out_dir, routine)

    ## check if save path exists
    if not os.path.isdir(routine_dir):
        os.makedirs(routine_dir, exist_ok = True)
    return routine_dir

################################################################################## DEBUGGING
################################################################################
def save_rep(model, d_config, m_config, sub_dir = None):
    """Save fitted model (with data!) to file.
    
    ARGS:
        model - an instance of Model class
    RAISES:
        None
    RETURNS:
        None
    """
    debug = model.data.debug

    ## get output directory path
    out_dir = output_directory_path(d_config, m_config)
    save_dir = os.path.join(
        out_dir,
        debug['routine'],
        f'iter{debug['iter']}'
    )

    if sub_dir is not None:
        save_dir = os.path.join(save_dir, sub_dir)

    ## check if save path exists
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok = True)

    with open(os.path.join(save_dir, 
        f'rep{debug['rep']}_model.pickle'), "wb") as f:
        pickle.dump(model, f)


# def debug_sample(model, d_config, m_config):
#     """Document the sampled data (helpful for diagnosing convergence/optimization problems).
    
#     ARGS:
#         model - an instance of Model class
#     RAISES:
#         None
#     RETURNS:
#         None
#     """
#     debug = model.data.debug

#     ## get output directory path
#     out_dir = output_directory_path(d_config, m_config)
#     save_dir = os.path.join(
#         out_dir,
#         debug['routine']
#     )
#     ## add subdirectories for iteration
#     save_dir = os.path.join(
#         save_dir,
#         f'iter{debug['iter']}'
#     )

#     ## check if save path exists
#     if not os.path.isdir(save_dir):
#         os.makedirs(save_dir, exist_ok = True)

#     with open(os.path.join(save_dir, 
#         f'rep{debug['rep']}_model.pickle'), "wb") as f:
#         pickle.dump(model, f)



# def debug_fit(model, fit, method = 'default'):
#     """Document the fitted glmm (helpful for diagnosing convergence/optimization problems).
    
#     ARGS:
#         model - an instance of Model class
#         fit   - glmmTMB fitted model
#     RAISES:
#         None
#     RETURNS:
#         None
#     """
#     ## imports
#     base   = importr('base')
#     xtable = importr('xtable')
#     broom_mix = importr('broom.mixed')
#     unlist = ro.r('unlist')
#     class_type = ro.r('class')
#     is_null = ro.r('is.null')
#     as_list = ro.r('as.list')

#     ## debugging dictionary from model
#     debug = model.data.debug

#     ## get output directory path
#     out_dir = output_directory_path(data = model.data)
#     save_dir = os.path.join(
#         out_dir,
#         debug['routine']
#     )
#     if model.config['data']['outcome'] != 'real':
#         save_dir = os.path.join(
#             save_dir,
#             f'iter{debug['iter']}',
#             f'rep{debug['rep']}'
#         )
#     if not os.path.isdir(save_dir):
#         os.makedirs(save_dir)

#     ## "dynamic" naming
#     label_px = f'conv_err_{method}'
#     counter = 1
#     label = label_px + '_' + str(counter)
#     while os.path.isdir(os.path.join(save_dir, label)):
#         counter += 1
#         label   = label_px + '_' + str(counter)

#     file_name = os.path.join(save_dir, f'{label}.txt')

#     ## extract summary
#     fit_summary = base.summary(fit)

#     ### extract relevant info
#     call = f'Model: \n {fit_summary.rx2('call')} \n'
#     log_lik = f'Log-Likelihood: \n \t {fit_summary.rx2('logLik')[0]} \n'
#     fam  = f'Family: \n \t {fit_summary.rx2('family')[0]} \n'
#     link = f'Link: \n \t {fit_summary.rx2('link')[0]} \n'

#     if model.config['model']['outcome'] == 'neg_bin':
#         sigma = f'Gamma: \n \t {fit_summary.rx2('sigma')[0]} \n'
#     else:
#         sigma = f'Phi: \n \t {fit_summary.rx2('sigma')[0]} \n'

#     ## random effects variances
#     ## see pg. 14 https://cran.r-project.org/web/packages/glmmTMB/vignettes/model_evaluation.pdf
#     tau_sqr = base.unname(unlist(fit_summary.rx('varcor')))
#     var_corr = f'Tau^2: {tau_sqr} \n'

#     # ## see pg. 13 https://cran.r-project.org/web/packages/glmmTMB/vignettes/model_evaluation.pdf
#     # fit.do_slot_assign('class', 'TMB')

#     with open(file_name, 'w') as f:
#         f.write(fam)
#         f.write(link)
#         f.write(log_lik)
#         f.write(sigma)
#         if not is_null(tau_sqr):
#             f.write(var_corr)
#         f.write(call)


# def debug_optim(model):
#     """Document the passed arguments to the optimization scheme (cone program). 
    
#     ARGS:
#         model - an instance of Model class
#     RAISES:
#         None
#     RETURNS:
#         None
#     """
#     ## debugging dictionary from model
#     debug = model.data.debug

#     ## get output directory path
#     out_dir = output_directory_path(data = model.data)
#     save_dir = os.path.join(
#         out_dir,
#         debug['routine'],
#         f'iter{debug['iter']}',
#         f'rep{debug['rep']}'
#     )

#     ## check if save path exists
#     if not os.path.isdir(save_dir):
#         os.makedirs(save_dir)

#     ## compute score
#     score_val, scale_val = score.compute_score_var_comp(
#         model = model, 
#         scale = model.config['model']['scale']
#     )
#     ## compute information
#     info_val = info.compute_info(
#         model     = model, 
#         scale_val = scale_val
#     )
#     ## schur complement
#     schur_inv = helpers.lower_right_inv(
#         mat = info_val,
#         q   = model.q
#     )

#     ## intermediate terms
#     omega     = intermed.compute_omega(model)
#     delta_inv = intermed.compute_delta(model, inv = True)
#     epsilon   = intermed.compute_epsilon(model)
#     xi        = intermed.compute_xi(model)
#     c         = intermed.compute_c(model), 
#     kappa2    = model.compute_kappa(order = 2)
#     kappa3    = model.compute_kappa(order = 3)
#     kappa4    = model.compute_kappa(order = 4)

#     ## save
#     jnp.savez(
#         os.path.join(save_dir, 'cone_program.npz'),
#         scale_val = scale_val,
#         score_val = score_val,
#         info_val  = info_val,
#         schur_inv = schur_inv
#     )
#     jnp.savez(
#         os.path.join(save_dir, 'intermediate.npz'),
#         omega     = omega,
#         delta_inv = delta_inv,
#         epsilon   = epsilon,
#         xi        = xi, 
#         c         = c,
#         kappa2    = kappa2,
#         kappa3    = kappa3,
#         kappa4    = kappa4
#     )

