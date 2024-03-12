from .layer import *

from typing import Dict
import torch_geometric as pyg


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


def common_graph_layer(layer):
    class Wrapper(BaseWrapper):

        def __init__(self, configuration):
            super().__init__(configuration)

            self.model = layer(**configuration)

        @property
        def signature(self):
            return self._signature or 'x, edge_index -> x'

        def forward(self, x, edge_index):
            return self.model(x, edge_index)

    return Wrapper


def common_operation(fn):
    class Wrapper(BaseWrapper):

        @property
        def signature(self):
            return self._signature or 'x, batch -> x'

        def forward(self, x, batch):
            if isinstance(batch, tuple):
                return fn(x, batch=[element for element in batch if element.shape[0] == x.shape[0]][0])

            return fn(x, batch=batch)

    return Wrapper
