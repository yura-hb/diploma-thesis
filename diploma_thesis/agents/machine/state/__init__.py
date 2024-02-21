from functools import partial

from utils import from_cli
from .deep_marl_indirect import DEEPMARLIndirectStateEncoder
from .deep_marl_mr import DEEPMARLMinimumRepetitionStateEncoder
from .encoder import StateEncoder
from .plain import PlainEncoder

key_to_class = {
    "plain": PlainEncoder,
    "deep_marl_mr": DEEPMARLMinimumRepetitionStateEncoder,
    "deep_marl_indirect": DEEPMARLIndirectStateEncoder
}

from_cli = partial(from_cli, key_to_class=key_to_class)
