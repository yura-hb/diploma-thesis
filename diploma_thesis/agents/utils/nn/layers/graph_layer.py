from typing import Dict

import torch_geometric as pyg

from .layer import *


class GraphLayer(Layer):

    @property
    @abstractmethod
    def signature(self) -> str | None:
        pass


class BaseWrapper(GraphLayer):

    def __init__(self, configuration):
        super().__init__()

        self.configuration = configuration

        if 'signature' in configuration:
            self._signature = configuration['signature']

            del configuration['signature']
        else:
            self._signature = None

    @property
    def signature(self):
        return self._signature or 'x -> x'

    @classmethod
    def from_cli(cls, parameters: Dict):
        return cls(parameters)


class GraphModuleWrapper(BaseWrapper):

    def __init__(self, layer, configuration):
        super().__init__(configuration)

        self.model = layer(**configuration)

    @property
    def signature(self):
        return self._signature or 'x, edge_index -> x'

    def forward(self, x, edge_index):
        return self.model(x, edge_index)


class GIN(GraphModuleWrapper):

    def __init__(self, configuration):
        super().__init__(pyg.nn.GIN, configuration)


class SAGEConv(GraphModuleWrapper):

    def __init__(self, configuration):
        super().__init__(pyg.nn.SAGEConv, configuration)


class GATConv(GraphModuleWrapper):

    def __init__(self, configuration):
        super().__init__(pyg.nn.GATConv, configuration)


class GCNConv(GraphModuleWrapper):

    def __init__(self, configuration):
        super().__init__(pyg.nn.GCNConv, configuration)


class DeepGCNConv(GraphModuleWrapper):

    def __init__(self, configuration):
        super().__init__(pyg.nn.DeepGCNLayer, configuration)


class GraphFunctionWrapper(BaseWrapper):

    def __init__(self, fn, configuration):
        super().__init__(configuration)

        self.fn = fn

    @property
    def signature(self):
        return self._signature or 'x, batch -> x'

    def forward(self, x, batch):
        if isinstance(batch, tuple):
            return self.fn(x, batch=[element for element in batch if element.shape[0] == x.shape[0]][0])

        return self.fn(x, batch=batch)


class AddPool(GraphFunctionWrapper):

    def __init__(self, configuration):
        super().__init__(pyg.nn.global_add_pool, configuration)


class MaxPool(GraphFunctionWrapper):

    def __init__(self, configuration):
        super().__init__(pyg.nn.global_max_pool, configuration)


class MeanPool(GraphFunctionWrapper):

    def __init__(self, configuration):
        super().__init__(pyg.nn.global_mean_pool, configuration)
