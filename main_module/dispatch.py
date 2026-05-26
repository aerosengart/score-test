"""
Dispatch dictionaries for construction of Data/Model.
"""
from data_module import neg_bin_data, splash_data, neg_bin_model
from main_module import routines

data_types = {
    'neg_bin': neg_bin_data.Data,
    'splash':  splash_data.Data,
}

model_types = {
    'neg_bin': neg_bin_model.Model,
}

routine_types = {
    'score_var_comp': routines.score_var_comp_empirical_dist,
    'one_two_sided': routines.one_two_sided_comparison,
}