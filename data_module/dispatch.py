"""
Dispatch dictionaries for construction of Data/Model.
"""
from data_module import neg_bin_data, splash_data, poi_data, alt_splice_data, ell_one_data

data_types = {
    'neg_bin': neg_bin_data.Data,
    'poi':     poi_data.Data,
    'splash':  splash_data.Data,
    'alt_splice': alt_splice_data.Data,
    'ell_one': ell_one_data.Data,
}

scenarios = ['alt_splice', 'ell_one']

