"""
Dispatch dictionaries for running modules.
"""
from main_module import routines, plot

routine_types = {
    'score_var_comp': routines.score_var_comp_emp_dist,
    'one_two_sided':  routines.one_two_sided_comparison,
    'fx_effs_sens':   routines.test_stat_fx_eff_sensitivity
}

plot_types = {
    'score_var_comp': plot.score_hist,
    'one_two_sided':  plot.one_two_sided_qq,
}