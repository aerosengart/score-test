"""
Dispatch dictionaries for running modules.
"""
from routine_module import emp_dists, error_comps, power_comps, test_stat_comps, test_stat_sens
import numpy as np

routine_types = {
    'score_var_comp':   emp_dists.score_var_comp_emp_dist,
    'one_two_sided':    test_stat_comps.one_two_sided_comparison,
    'fx_effs_sens':     test_stat_sens.test_stat_fx_eff_sensitivity,
    'power_by_var_comp': power_comps.power_by_var_comp,
    'power_by_n_samps':  power_comps.power_by_n_samps,
    'power_by_n_targs':  power_comps.power_by_n_targs,
}

plot_types = {
    'score_var_comp':    emp_dists.score_hist,
    'one_two_sided':     test_stat_comps.one_two_sided_qq,
    'fx_effs_sens':      test_stat_sens.fx_effs_jitter,
    'power_by_var_comp': power_comps.plot_power_by_tau_sqr,
    'power_by_n_samps':  power_comps.plot_power_by_n_samps,
    'power_by_n_targs':  power_comps.plot_power_by_n_targs,
}