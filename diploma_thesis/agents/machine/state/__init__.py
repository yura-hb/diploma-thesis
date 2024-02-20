from typing import Dict

from .deep_marl_indirect import DEEPMARLIndirectStateEncoder
from .deep_marl_mr import DEEPMARLMinimumRepetitionStateEncoder
from .encoder import StateEncoder
from .plain import PlainEncoder

key_to_class = {
    "plain": PlainEncoder,
    "deep_marl_mr": DEEPMARLMinimumRepetitionStateEncoder,
    "deep_marl_indirect": DEEPMARLIndirectStateEncoder
}


def from_cli(parameters: Dict) -> StateEncoder:
    cls = key_to_class[parameters['kind']]

    return cls.from_cli(parameters.get('parameters', {}))
