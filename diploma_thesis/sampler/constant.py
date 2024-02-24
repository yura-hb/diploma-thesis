import torch

from .numeric_sampler import NumericSampler


class Constant(NumericSampler):

    def __init__(self, value: float):
        super().__init__()

        self.value = value

    def sample(self, shape):
        return torch.ones(shape) * self.value

    @property
    def mean(self):
        return torch.tensor(self.value)

    @property
    def variance(self):
        return torch.tensor(0.0)

    @staticmethod
    def from_cli(parameters):
        return Constant(parameters['value'])
