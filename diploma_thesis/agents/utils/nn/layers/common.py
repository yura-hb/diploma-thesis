import torch

from .layer import *
from typing import List


class Flatten(Layer):

    def __init__(self):
        super().__init__()

        self.layer = nn.Flatten()

    def forward(self, batch: torch.FloatTensor) -> torch.FloatTensor:
        return self.layer(batch)

    @classmethod
    def from_cli(cls, parameters: dict) -> 'Layer':
        return Flatten()


class LayerNorm(Layer):

    def __init__(self, normalized_shape: int | List[int]):
        super().__init__()

        self.layer = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, batch: torch.FloatTensor) -> torch.FloatTensor:
        return self.layer(batch)

    @classmethod
    def from_cli(cls, parameters: dict) -> 'Layer':
        return LayerNorm(normalized_shape=parameters['normalized_shape'])


class InstanceNorm(Layer):

    def __init__(self):
        super().__init__()

        self.layer = nn.LazyInstanceNorm1d()

    def forward(self, batch: torch.FloatTensor) -> torch.FloatTensor:
        return self.layer(batch)

    @classmethod
    def from_cli(cls, parameters: dict) -> 'Layer':
        return InstanceNorm()
