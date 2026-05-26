import matplotlib.pyplot as plt
import numpy             as np

from score_module import helpers

#plt.rcParams['text.usetex'] = True

## some sort of issue with matplotlib, latex, and the unicode minus sign???
plt.rcParams['axes.unicode_minus'] = False


def score_hist(scores_arr, dim, plt_name):
    if dim == 1:
        ## variance component is scalar-valued
        _, axs = plt.subplots(
            1, 1, 
            tight_layout = True,
            figsize = (10, 6)
        )
        axs.hist(scores_arr.tolist(), bins = 100, density = True)    
        axs.set_title(f'analytical score component under $H_0$')
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
            axs[i].set_title(f'analytical score component {i} under $H_0$')
            axs[i].set_xlabel('score')
            axs[i].set_ylabel('density')
    plt.savefig(plt_name)
    plt.close()






def lrt_score_qq(lrt_stats, score_stats, q, k, plt_name):
    """Save score/likelihood ratio test comparison."""
    plt.rcParams.update({'font.size': 8})
    fig, axs = plt.subplots(
        1, 2,
        tight_layout = True, 
        figsize = (12, 6)
    )

    iters = lrt_stats.shape[1]

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
        ## compute quantiles
        lrt_qs   = np.quantile(lrt_stats[:, i], quantiles)
        score_qs = np.quantile(score_stats[:, i], quantiles)
        
        ## plot two-sided stats
        axs[0].axline((0, 0), slope = 1, color = 'black', linestyle = 'dotted')
        axs[0].plot(chisqr_mix_qs, lrt_qs, color = 'red', alpha = 0.3)

        ## plot one-sided stats
        axs[1].axline((0, 0), slope = 1, color = 'black', linestyle = 'dotted')
        axs[1].plot(chisqr_mix_qs, score_qs, color = 'blue', alpha = 0.3)

    axs[0].set_title('Likelihood Ratio Test')
    axs[1].set_title('Score Test')
    axs[0].set_xlabel(r'$\bar{\chi}^2$ theoretical quantiles')
    axs[1].set_xlabel(r'$\bar{\chi}^2$ theoretical quantiles')
    axs[0].set_ylabel('empirical quantiles')
    axs[1].set_ylabel('empirical quantiles')
    plt.suptitle('QQ Plots of One-Sided Test Statistics')
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(plt_name)
    plt.close()


def one_two_sided_qq(one_sided_stats, two_sided_stats, q, k, plt_name):
    """Save one-/two-sided score test comparison."""
    plt.rcParams.update({'font.size': 8})
    _, axs = plt.subplots(
        1, 2,
        tight_layout = True, 
        figsize = (12, 6)
    )

    ## set labels
    axs[0].set_xlabel(r'$\chi^2$ quantiles')
    axs[0].set_ylabel('empirical quantiles')
    axs[0].set_title(f'Two-Sided Test Statistic')

    axs[1].set_xlabel(r'$\bar{\chi}^2$ quantiles')
    axs[1].set_ylabel('empirical quantiles')
    axs[1].set_title(f'One-Sided Test Statistic')

    iters = one_sided_stats.shape[-1]

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

    ## loop over columns
    for l in range(iters):
        ## compute quantiles
        one_sided_qs = np.quantile(one_sided_stats[:, l], quantiles)
        two_sided_qs = np.quantile(two_sided_stats[:, l], quantiles)
        
        ## plot two-sided stats
        axs[0].axline((0, 0), slope = 1, color = 'black', linestyle = 'dotted')
        axs[0].plot(chisqr_qs, two_sided_qs, color = 'red', alpha = 0.3)

        ## plot one-sided stats
        axs[1].axline((0, 0), slope = 1, color = 'black', linestyle = 'dotted')
        axs[1].plot(chisqr_mix_qs, one_sided_qs, color = 'blue', alpha = 0.3)

    plt.suptitle('QQ Plots of One-Sided and Two-Sided Test Statistic')
    plt.savefig(plt_name)
    plt.close()



def lrt_score_jitter_grid(lrt_stats, score_stats, q, k, x_axis_ticks, plt_name):
    """Save grid of QQ-plots with jitter."""
    plt.rcParams.update({'font.size': 8})
    fig, axs = plt.subplots(
        len(x_axis_ticks), 2,
        tight_layout = True, 
        figsize = (8, 24)
    )

    iters = lrt_stats.shape[0]

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
        for l in range(len(x_axis_ticks)):
            ## compute quantiles
            lrt_qs   = np.quantile(lrt_stats[i, :, l], quantiles)
            score_qs = np.quantile(score_stats[i, :, l], quantiles)
            
            ## plot two-sided stats
            axs[l, 0].axline((0, 0), slope = 1, color = 'black', linestyle = 'dotted')
            axs[l, 0].plot(chisqr_mix_qs, lrt_qs, color = 'red', alpha = 0.3)
            axs[l, 0].set_ylabel(f'Jitter = {x_axis_ticks[l]}')

            ## plot one-sided stats
            axs[l, 1].axline((0, 0), slope = 1, color = 'black', linestyle = 'dotted')
            axs[l, 1].plot(chisqr_mix_qs, score_qs, color = 'blue', alpha = 0.3)

    axs[0, 0].set_title('Likelihood Ratio Test')
    axs[0, 1].set_title('Score Test')
    axs[len(x_axis_ticks) - 1, 0].set_xlabel(r'$\bar{\chi}^2$ theoretical quantiles')
    axs[len(x_axis_ticks) - 1, 1].set_xlabel(r'$\bar{\chi}^2$ theoretical quantiles')
    plt.suptitle('QQ Plots of One-Sided Test Statistics with Jitter')
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(plt_name)
    plt.close()