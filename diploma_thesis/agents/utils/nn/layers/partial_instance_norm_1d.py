
import torch

from .layer import *
from torch import nn


class PartialInstanceNorm1d(Layer):

    def __init__(self, channels: int):
        super().__init__()

        self.channels = channels
        self.norm = nn.InstanceNorm1d(num_features=1)

    def forward(self, batch):
        normalized = batch[:, :self.channels]
        normalized = self.norm(normalized)

        return torch.hstack([normalized, batch[:, self.channels:]])

    @classmethod
    def from_cli(cls, parameters: dict) -> 'Layer':
        return PartialInstanceNorm1d(channels=parameters['channels'])
