
import yaml
from .dict import *


def modified(parameters):
    base_parameters = parameters['base_path']

    with open(base_parameters) as file:
        base_parameters = yaml.safe_load(file)

    mods = parameters['mods']

    for mod in mods:
        with open(mod) as file:
            mod = yaml.safe_load(file)
            base_parameters = merge_dicts(base_parameters, mod)

    return base_parameters
