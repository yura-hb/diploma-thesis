
from .modified import modified


def from_cli(key_to_cls, parameters, *args, **kwargs):
    kind = parameters['kind']

    if kind == 'mod':
        parameters = modified(parameters)
        return from_cli(parameters, *args, **kwargs)

    cls = key_to_cls[parameters['kind']]

    return cls.from_cli(parameters['parameters'], *args, **kwargs)