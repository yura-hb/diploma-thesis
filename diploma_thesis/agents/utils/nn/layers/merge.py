
import torch

from .layer import *


class Merge(Layer):

    def __init__(self, kind: str = 'pass'):
        super().__init__()

        self.kind = kind

    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        match self.kind:
            case 'concat':
                return torch.cat((lhs, rhs), dim=1)
            case 'add':
                return lhs + rhs
            case 'mult':
                return lhs * rhs
            case 'pass':
                if lhs is not None:
                    return lhs

                if rhs is not None:
                    return rhs

                raise ValueError("Both inputs are None")
            case _:
                raise ValueError(f"Unknown merge function {self.kind}")

    @classmethod
    def from_cli(cls, parameters: dict) -> 'Layer':
        return Merge(kind=parameters['kind'])
