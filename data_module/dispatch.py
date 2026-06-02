"""
Dispatch dictionaries for construction of Data/Model.
"""
from data_module import neg_bin_data, splash_data, oasis_data, neg_bin_model, poi_data, multinom_data

data_types = {
    'neg_bin': neg_bin_data.Data,
    'poi':     poi_data.Data,
    'multinom': multinom_data.Data,
    'splash':  splash_data.Data,
    'oasis':   oasis_data.Data,
}

model_types = {
    'neg_bin': neg_bin_model.Model,
}