from functools import partial

from utils import from_cli
from .deep_marl_indirect import DEEPMARLIndirectStateEncoder
from .deep_marl_mr import DEEPMARLMinimumRepetitionStateEncoder
from .djsp_graph_encoder import DJSPGraphEncoder
from .encoder import StateEncoder
from .auxiliary_graph_encoder import AuxiliaryGraphEncoder
from .hierarchical_graph_encoder import HierarchicalGraphEncoder
from .plain import PlainEncoder
from .vp_graph_encoder import VPTGraphEncoder
from .custom_encoder import CustomGraphEncoder
from .custom_encoder_v2 import CustomV2GraphEncoder

key_to_class = {
    "plain": PlainEncoder,
    "deep_marl_mr": DEEPMARLMinimumRepetitionStateEncoder,
    "deep_marl_indirect": DEEPMARLIndirectStateEncoder,
    'auxiliary': AuxiliaryGraphEncoder,
    'djsp': DJSPGraphEncoder,
    'hierarchical': HierarchicalGraphEncoder,
    'vpt': VPTGraphEncoder,
    'custom': CustomGraphEncoder,
    'custom_v2': CustomV2GraphEncoder
}

from_cli = partial(from_cli, key_to_class=key_to_class)
