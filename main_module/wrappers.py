import yaml

from main_module import dispatch


def run_routine(d_config, m_config, routine_name, **kwargs):
    ## read config files as dictionaries
    with open(d_config, 'r') as file:
        d_dict = yaml.safe_load(file)
    with open(m_config, 'r') as file:
        m_dict = yaml.safe_load(file)
    dispatch.routine_types[routine_name](d_dict, m_dict, **kwargs)