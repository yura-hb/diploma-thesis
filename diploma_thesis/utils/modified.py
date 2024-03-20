
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
    mods = reduce(lambda x, y: x + y if isinstance(y, list) else x + [y], mods, [])

    mods_dir = os.path.dirname(base_path)
    mods_dir = os.path.join(mods_dir, 'mods')
    mod_dirs = parameters.get('mod_dirs', []) + [mods_dir]

    base_parameters = __fetch_mods__(base_parameters, mods, mod_dirs)

    for key, value in template.items():
        if key in base_parameters:
            template[key] = base_parameters[key]

    base_parameters = __apply_template__(base_parameters, template)

    if 'nested' in parameters:
        base_parameters = merge_dicts(base_parameters, parameters['nested'])

    return base_parameters


def __fetch_mods__(base_parameters, mods, dirs):
    for mod in mods:
        did_found_mod = False

        for directory in dirs:
            if len(mods) == 0:
                break

            path = os.path.join(directory, mod)

            if not os.path.exists(path):
                continue

            did_found_mod = True

            with open(path) as file:
                mod = yaml.safe_load(file)
                base_parameters = merge_dicts(base_parameters, mod)

            break

        if not did_found_mod:
            raise ValueError(f'Mod does not exist {mod} in {dirs}')

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
