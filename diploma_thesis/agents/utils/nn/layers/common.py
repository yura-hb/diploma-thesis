import torch

from .layer import *


class Flatten(Layer):

    def __init__(self):
        super().__init__()

        self.layer = nn.Flatten()

    def __call__(self, batch: torch.FloatTensor) -> torch.FloatTensor:
        return self.layer(batch)

    @classmethod
    def from_cli(cls, parameters: dict) -> 'Layer':
        return Flatten()


# class LayerNorm(Layer):
#
#     def __init__(self):
#         super().__init__()
#
#         self.layer = nn.LayerNorm()
#
#
#     def __call__(self, batch: torch.FloatTensor) -> torch.FloatTensor:
#         return self.layer(batch)
#
#
#     @classmethod
#     def from_cli(cls, parameters: dict) -> 'Layer':
#         return LayerNorm(dimension=parameters['dimension'])

class InstanceNorm(Layer):

    def __init__(self):
        super().__init__()

        self.layer = nn.LazyInstanceNorm1d()

    def __call__(self, batch: torch.FloatTensor) -> torch.FloatTensor:
        return self.layer(batch)

    @classmethod
    def from_cli(cls, parameters: dict) -> 'Layer':
        return InstanceNorm()
