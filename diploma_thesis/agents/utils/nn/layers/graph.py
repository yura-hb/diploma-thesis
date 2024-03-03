
from .layer import *

import torch_geometric as pyg


class GraphLayer(Layer):

    def __init__(self, kind: str):
        super().__init__()

        self.kind = kind
        self.layer = self.__build__()

    def __build__(self):
        match self.kind:
            case 'SageConv':
                return pyg.nn.SAGEConv(out_channels=1, in_channels=1)
            case _:
                raise ValueError(f"Unknown graph layer {self.kind}")

    @classmethod
    def from_cli(cls, parameters: dict) -> 'Layer':
        pass

