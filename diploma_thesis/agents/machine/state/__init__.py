import logging

from typing import Dict

from .encoder import StateEncoder
from .plain import PlainEncoder
from .deep_marl_direct import DEEPMARLDirectStateEncoder
from .deep_marl_indirect import DEEPMARLIndirectStateEncoder

key_to_class = {
    "plain": PlainEncoder,
    "deep_marl_direct": DEEPMARLDirectStateEncoder,
    "deep_marl_indirect": DEEPMARLIndirectStateEncoder
}


def from_cli(parameters: Dict) -> StateEncoder:
    cls = key_to_class[parameters['kind']]

    return cls.from_cli(parameters.get('parameters', {}))
