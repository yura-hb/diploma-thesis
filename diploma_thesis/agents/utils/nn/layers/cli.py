from .graph_layer import *


def from_cli(*args, **kwargs):
    from .linear import Linear
    from .common import Flatten, InstanceNorm, LayerNorm
    from .activation import Activation
    from .merge import Merge
    from .graph_model import GraphModel
    from .partial_instance_norm_1d import PartialInstanceNorm1d
    from .output import Output

    from utils import from_cli as from_cli_

    key_to_class = {
        'flatten': Flatten,
        'merge': Merge,
        'activation': Activation,

        'linear': Linear,
        'layer_norm': LayerNorm,
        'instance_norm': InstanceNorm,
        'partial_instance_norm': PartialInstanceNorm1d,

        'graph_model': GraphModel,
        'gin': GIN,
        'sage': SAGEConv,
        'gat': GATConv,
        'gcn': GCNConv,
        'deep_gcn': DeepGCNConv,

        'add_pool': AddPool,
        'max_pool': MaxPool,
        'mean_pool': MeanPool,

        'output': Output
    }

    return from_cli_(*args, **kwargs, key_to_class=key_to_class)
