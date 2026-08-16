"""
Dispatch dictionaries for running modules.
"""
from routine_module import emp_dists, error_comps, stat_comps, stat_sens, one_two_sided_fig, error_comp_fig, sensitivity_fig
import numpy as np

routine_types = {
    'score_var_comp':    emp_dists.score_var_comp_emp_dist,
    'one_two_sided':     stat_comps.one_two_sided_comparison,
    'score_lrt_comp':    stat_comps.score_lrt_comparison,
    'fx_effs_sens':      stat_sens.test_stat_sensitivity,
    'disp_sens':         stat_sens.test_stat_sensitivity,
    'err_comp':          error_comps.error_comparison,
    'one_two_sided_fig': one_two_sided_fig.one_two_sided_figure,
    'err_comp_fig':      error_comp_fig.error_comparison,
    'sensitivity_fig':   sensitivity_fig.test_stat_sensitivity,
}

plot_types = {
    'score_var_comp':    emp_dists.score_hist,
    'score_var_comp_by_disp': emp_dists.score_var_comp_by_disp_emp_dist,
    'one_two_sided':     stat_comps.plot_one_two_sided,
    'score_lrt_comp':    stat_comps.plot_score_lrt,
    'fx_effs_sens':      stat_sens.qq_jitter_grid,
    'disp_sens':         stat_sens.qq_jitter_grid,
    'err_comp':          error_comps.plot_error_comparison,
    'one_two_sided_fig': one_two_sided_fig.plot_one_two_sided,
    'err_comp_fig':      error_comp_fig.plot_error_comparison,
    'sensitivity_fig':   sensitivity_fig.plot_sensitivity,
}