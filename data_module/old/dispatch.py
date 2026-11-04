"""
Dispatch dictionaries for construction of Data/Model.
"""
from data_module import dom_target, neg_bin_data, splash_data, poi_data
from data_module.todo import ell_one_data

data_types = {
    'neg_bin': neg_bin_data.Data,
    'poi':     poi_data.Data,
    'splash':  splash_data.Data,
    'alt_splice': dom_target.Data,
    'ell_one': ell_one_data.Data,
}

scenarios = ['alt_splice', 'ell_one']

