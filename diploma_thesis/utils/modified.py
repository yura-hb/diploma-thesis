
import os
import yaml
from .dict import *


def modified(parameters):
    base_path = parameters['base_path']

    with open(base_path) as file:
        base_parameters = yaml.safe_load(file)

    template = dict()

    if 'template' in parameters:
        template = __load_template__(parameters, base_path)

    mods = parameters['mods']

    mods_dir = os.path.dirname(base_path)
    mods_dir = os.path.join(mods_dir, 'mods')

    for mod in mods:
        with open(os.path.join(mods_dir, mod)) as file:
            mod = yaml.safe_load(file)
            base_parameters = merge_dicts(base_parameters, mod)

    base_parameters = __apply_template__(base_parameters, template)

    return base_parameters


def __load_template__(parameters, base_path):
    template_path = os.path.dirname(base_path)
    template_path = os.path.join(template_path, 'templates', parameters['template'])

    values = dict()

    for file in os.listdir(template_path):
        if file.endswith('.yml'):
            with open(os.path.join(template_path, file)) as f:
                template = yaml.safe_load(f)
                values[os.path.basename(file).split('.')[0]] = template

    values = {f'__{k}__': v for k, v in values.items()}

    return values


def __apply_template__(parameters, template):
    if isinstance(parameters, dict):
        updates = dict()

        for k, v in parameters.items():
            if k in template:
                updates[k] = template[k]
            else:
                parameters[k] = __apply_template__(v, template)

        for k, v in updates.items():
            del parameters[k]
            parameters.update(v)

    if isinstance(parameters, list):
        for i, v in enumerate(parameters):
            parameters[i] = __apply_template__(v, template)

    return parameters
