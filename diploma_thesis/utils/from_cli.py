
from .modified import modified
from .persistence import load


def from_cli(parameters, key_to_class, return_params: bool = False, *args, **kwargs):

    def wrap(value, params):
        if return_params:
            return value, params

        return value

    kind = parameters['kind']
    parameters = parameters.get('parameters', {})

    if kind == 'mod':
        parameters = modified(parameters)
        return wrap(from_cli(parameters, key_to_class=key_to_class, *args, **kwargs), parameters)

    if kind == 'persisted':
        return wrap(load(parameters['path']), {})

    cls = key_to_class[kind]

    return wrap(cls.from_cli(parameters, *args, **kwargs), parameters)