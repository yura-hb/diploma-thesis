
import os
import yaml
from .dict import *


def modified(parameters):
    base_path = parameters['base_path']

    with open(base_path) as file:
        base_parameters = yaml.safe_load(file)

    mods = parameters['mods']

    mods_dir = os.path.dirname(base_path)
    mods_dir = os.path.join(mods_dir, 'mods')

    for mod in mods:
        with open(os.path.join(mods_dir, mod)) as file:
            mod = yaml.safe_load(file)
            base_parameters = merge_dicts(base_parameters, mod)

    return base_parameters
