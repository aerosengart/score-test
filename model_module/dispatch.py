"""
Dispatch dictionaries for construction of Data/Model.
"""
from model_module import neg_bin_model

model_types = {
    'neg_bin': neg_bin_model.Model,
}