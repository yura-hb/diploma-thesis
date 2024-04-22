from .graph_layer import *


def from_cli(*args, **kwargs):
    from .alias import Alias
    from .linear import Linear
    from .common import Flatten, InstanceNorm, LayerNorm, BatchNorm1d
    from .activation import Activation
    from .merge import Merge
    from .merge_graph import MergeGraph
    from .shared import Shared
    from .graph_model import GraphModel
    from .partial_instance_norm_1d import PartialInstanceNorm1d
    from .recurrent import Recurrent
    from .output import Output
    from utils import from_cli as from_cli_

    key_to_class = {
        'flatten': Flatten,
        'merge': Merge,
        'merge_graph': MergeGraph,
        'activation': Activation,

        'linear': Linear,
        'layer_norm': LayerNorm,
        'batch_norm_1d': BatchNorm1d,
        'instance_norm': InstanceNorm,
        'partial_instance_norm': PartialInstanceNorm1d,

        'graph_model': GraphModel,
        'gin': GIN,
        'sage': SAGEConv,
        'gat': GATConv,
        'gcn': GCNConv,
        'deep_gcn': DeepGCNConv,
        'graph_instance_norm': GraphInstanceNorm,
        'mask': MaskTarget,
        'attention': GraphAttention,

        'add_pool': AddPool,
        'max_pool': MaxPool,
        'mean_pool': MeanPool,
        'sag_pool': SAGPool,

        'alias': Alias,
        'select_target': SelectTarget,
        'shared': Shared,
        'recurrent': Recurrent,
        'output': Output
    }

    return from_cli_(*args, **kwargs, key_to_class=key_to_class)
