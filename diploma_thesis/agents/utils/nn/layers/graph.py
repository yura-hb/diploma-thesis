
from .layer import *
from typing import Dict

import torch_geometric as pyg


class GraphLayer(Layer):

    def __init__(self, kind: str, parameters: Dict):
        super().__init__()

        self.kind = kind
        self.layer = self.__build__()

    def __build__(self):
        match self.kind:
            case 'SageConv':
                return pyg.nn.SAGEConv(out_channels=1, in_channels=1)
            case _:
                raise ValueError(f"Unknown graph layer {self.kind}")

    def forward(self, data: pyg.data.Data) -> pyg.data.Data:
        return self.layer(data)

    @classmethod
    def from_cli(cls, parameters: dict) -> 'Layer':
        return cls(parameters['kind'], parameters=parameters)

