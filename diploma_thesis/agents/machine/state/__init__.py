from functools import partial

from utils import from_cli
from .deep_marl_indirect import DEEPMARLIndirectStateEncoder
from .deep_marl_mr import DEEPMARLMinimumRepetitionStateEncoder
from .djsp_graph_encoder import DJSPGraphEncoder
from .encoder import StateEncoder
from .auxiliary_graph_encoder import AuxiliaryGraphEncoder
from .plain import PlainEncoder

key_to_class = {
    "plain": PlainEncoder,
    "deep_marl_mr": DEEPMARLMinimumRepetitionStateEncoder,
    "deep_marl_indirect": DEEPMARLIndirectStateEncoder,
    'auxiliary': AuxiliaryGraphEncoder,
    'djsp': DJSPGraphEncoder,
    'hierarchical': ...,
    'vp': ...
}

from_cli = partial(from_cli, key_to_class=key_to_class)
