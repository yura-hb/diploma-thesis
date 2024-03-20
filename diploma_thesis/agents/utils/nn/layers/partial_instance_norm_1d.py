
import torch

from .layer import *
from torch import nn


class PartialInstanceNorm1d(Layer):

    def __init__(self, channels: int, signature: str):
        super().__init__(signature=signature)

        self.channels = channels
        self.norm = nn.InstanceNorm1d(num_features=1)

    def forward(self, batch):
        batch = torch.atleast_2d(batch)

        normalized = batch[:, :self.channels]
        normalized = self.norm(normalized)

        return torch.hstack([normalized, batch[:, self.channels:]])

    @classmethod
    def from_cli(cls, parameters: dict) -> 'Layer':
        return PartialInstanceNorm1d(channels=parameters['channels'], signature=parameters['signature'])
