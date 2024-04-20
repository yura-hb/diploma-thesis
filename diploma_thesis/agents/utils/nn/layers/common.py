import torch

from .layer import *
from typing import List


class Flatten(Layer):

    def __init__(self, signature):
        super().__init__(signature)

        self.layer = nn.Flatten()

    def forward(self, batch: torch.FloatTensor) -> torch.FloatTensor:
        return self.layer(batch)

    @classmethod
    def from_cli(cls, parameters: dict) -> 'Layer':
        return Flatten(parameters['signature'])


class LayerNorm(Layer):

    def __init__(self, normalized_shape: int | List[int], signature):
        super().__init__(signature)

        self.layer = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, batch: torch.FloatTensor) -> torch.FloatTensor:
        return self.layer(batch)

    @classmethod
    def from_cli(cls, parameters: dict) -> 'Layer':
        return LayerNorm(normalized_shape=parameters['normalized_shape'], signature=parameters['signature'])


class BatchNorm1d(Layer):

    def __init__(self, signature):
        super().__init__(signature)

        self.layer = nn.LazyBatchNorm1d()

    def forward(self, batch: torch.FloatTensor) -> torch.FloatTensor:
        return self.layer(batch)

    @classmethod
    def from_cli(cls, parameters: dict) -> 'Layer':
        return BatchNorm1d(signature=parameters['signature'])


class InstanceNorm(Layer):

    def __init__(self, signature):
        super().__init__(signature)

        self.layer = nn.LazyInstanceNorm1d()

    def forward(self, batch: torch.FloatTensor) -> torch.FloatTensor:
        return self.layer(batch)

    @classmethod
    def from_cli(cls, parameters: dict) -> 'Layer':
        return InstanceNorm(signature=parameters['signature'])
