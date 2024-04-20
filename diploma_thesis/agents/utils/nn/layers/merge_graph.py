
import torch

from .layer import *
from agents.base import Graph


class MergeGraph(Layer):

    def __init__(self, kind: str, signature: str):
        super().__init__(signature=signature)

        self.kind = kind

    def forward(self, batch: torch.Tensor, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        match self.kind:
            case 'concat':
                expanded = rhs[batch]

                return torch.cat((lhs, expanded), dim=1)
            case _:
                raise ValueError(f"Unknown merge function {self.kind}")

    @classmethod
    def from_cli(cls, parameters: dict) -> 'Layer':
        return MergeGraph(kind=parameters['kind'], signature=parameters['signature'])
