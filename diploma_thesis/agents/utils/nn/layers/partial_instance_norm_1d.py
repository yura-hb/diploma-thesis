
import torch

from torch import nn


class PartialInstanceNorm1d(nn.Module):

    def __init__(self, channels: int):
        super().__init__()

        self.channels = channels
        self.norm = nn.InstanceNorm1d(num_features=channels)

    def __call__(self, batch):
        normalized = batch[:, :self.channels]
        normalized = self.norm(normalized)

        return torch.hstack([normalized, batch[:, self.channels:]])
