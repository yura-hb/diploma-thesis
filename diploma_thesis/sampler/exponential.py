
from .numeric_sampler import NumericSampler

import torch


class Exponential(NumericSampler):

    def __init__(self, mean: float):
        super().__init__()
        self.distribution = torch.distributions.Exponential(rate=1 / mean)

    def sample(self, shape):
        return self.distribution.sample(shape)

    @property
    def mean(self):
        return self.distribution.mean

    @property
    def variance(self):
        return self.distribution.variance

    @staticmethod
    def from_cli(parameters):
        return Exponential(parameters['mean'])
