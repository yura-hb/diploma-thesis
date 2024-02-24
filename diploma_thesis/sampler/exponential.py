
from .numeric_sampler import NumericSampler

import torch


class Exponential(NumericSampler):

    def __init__(self, mean: float):
        super().__init__()

        self._mean = torch.tensor(mean)
        self.rate = torch.tensor(1 / mean)

    def sample(self, shape):
        return torch.zeros(shape).exponential_(lambd=self.rate, generator=self.generator)

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self.mean ** 2
    @staticmethod
    def from_cli(parameters):
        return Exponential(parameters['mean'])
