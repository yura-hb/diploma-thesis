
def from_cli(*args, **kwargs):
    from .linear import Linear
    from .common import Flatten, InstanceNorm, LayerNorm
    from .activation import Activation
    from .merge import Merge
    from .graph_model import GraphModel
    from .graph_layer import common_graph_layer, common_operation
    from .partial_instance_norm_1d import PartialInstanceNorm1d

    from utils import from_cli as from_cli_

    import torch_geometric as pyg

    key_to_class = {
        'flatten': Flatten,
        'merge': Merge,
        'activation': Activation,

        'linear': Linear,
        'layer_norm': LayerNorm,
        'instance_norm': InstanceNorm,
        'partial_instance_norm': PartialInstanceNorm1d,

        'graph_model': GraphModel,
        'gin': common_graph_layer(pyg.nn.GIN),
        'sage': common_graph_layer(pyg.nn.SAGEConv),
        'gat': common_graph_layer(pyg.nn.GATConv),
        'gcn': common_graph_layer(pyg.nn.GCNConv),
        'deep_gcn': common_graph_layer(pyg.nn.DeepGCNLayer),

        'add_pool': common_operation(pyg.nn.global_add_pool),
        'max_pool': common_operation(pyg.nn.global_max_pool),
        'mean_pool': common_operation(pyg.nn.global_mean_pool),
    }

    return from_cli_(*args, **kwargs, key_to_class=key_to_class)
