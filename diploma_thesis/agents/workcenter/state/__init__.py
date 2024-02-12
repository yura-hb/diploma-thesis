import logging

from .encoder import StateEncoder
from .plain import PlainEncoder


key_to_class = {
    "plain": PlainEncoder
}


def from_cli(parameters) -> StateEncoder:
    cls = key_to_class[parameters['kind']]

    return cls.from_cli(parameters.get('parameters', {}))
