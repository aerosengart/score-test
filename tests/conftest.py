import pytest
import yaml

from data_module.dispatch import data_types, model_types

## fixtures 
@pytest.fixture
def simul_data():
    def _method(path):
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        data_obj = data_types[config['outcome']](config)
        return data_obj
    return _method

@pytest.fixture
def splash_data():
    def _method(path, anchor):
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        data_obj = data_types[config['outcome']](config, anchor)
        return data_obj
    return _method

@pytest.fixture
def nb_model():
    def _method(path, data):
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        model_obj = model_types[config['outcome']](config, data)
        return model_obj
    return _method