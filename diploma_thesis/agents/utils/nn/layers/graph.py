
from .layer import *
from typing import Dict

import torch_geometric as pyg

# if len(self.configuration.graph) > 0:
#     self.graph_encoder = pyg_nn.Sequential('x, edge_index', [
#         (layer, layer.signature) if isinstance(layer, GraphLayer) else layer
#         for layer in self.configuration.graph
#     ])
# else:
#     self.graph_encoder = None

#
# if not self.is_configured:
#     if isinstance(data, HeteroData):
#         self.graph_encoder = to_hetero(self.graph_encoder, data.metadata())
#
# encoded_graph = self.graph_encoder(data.x_dict, data.edge_index_dict)


class GraphLayer(Layer):

    def __init__(self, configuration: Dict):
        super().__init__()

        self.kind = kind
        self.configuration = configuration
        self.layer = self.__build__()

    def __build__(self):
        match self.kind:
            case 'SageConv':
                return pyg.nn.SAGEConv(in_channels=-1, **self.configuration)
            case 'GIN':
                return pyg.nn.GIN(in_channels=-1, **self.configuration)
            case 'GAT':
                return pyg.nn.GAT(in_channels=-1, **self.configuration)
            case _:
                raise ValueError(f"Unknown graph layer {self.kind}")

    @property
    def signature(self):
        match self.kind:
            case 'SageConv' | 'GIN' | 'GAT':
                return 'x, edge_index -> x'
            case _:
                raise ValueError(f"Unknown graph layer {self.kind}")

    def forward(self, x, edge_index) -> pyg.data.Data:
        return self.layer(x, edge_index)

    @classmethod
    def from_cli(cls, parameters: dict) -> 'Layer':
        return cls(parameters['kind'], configuration=parameters['parameters'])

