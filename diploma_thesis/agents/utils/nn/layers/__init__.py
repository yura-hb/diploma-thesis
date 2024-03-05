
from .layer import Layer

from .linear import Linear
from .common import Flatten, InstanceNorm, LayerNorm
from .activation import Activation
from .merge import Merge
from .graph import GraphLayer
from .partial_instance_norm_1d import PartialInstanceNorm1d

from utils import from_cli
from functools import partial

key_to_class = {
    'linear': Linear,
    'flatten': Flatten,
    'activation': Activation,
    'layer_norm': LayerNorm,
    'instance_norm': InstanceNorm,
    'partial_instance_norm': PartialInstanceNorm1d,
    'noisy_linear': ...,
    'graph': GraphLayer,
    'merge': merge
}

from_cli = partial(from_cli, key_to_class=key_to_class)

