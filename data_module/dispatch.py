"""
Dispatch dictionaries for construction of Data/Model.
"""
from data_module import neg_bin_data, splash_data, neg_bin_model

data_types = {
    'neg_bin': neg_bin_data.Data,
    'splash':  splash_data.Data,
}

model_types = {
    'neg_bin': neg_bin_model.Model,
}