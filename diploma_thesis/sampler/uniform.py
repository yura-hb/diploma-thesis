
import torch

from typing import Tuple
from .numeric_sampler import NumericSampler


class Uniform(NumericSampler):

    def __init__(self, uniform: Tuple, noise: Tuple):
        super().__init__()

        assert uniform[0] < uniform[1], "Low bound of uniform sampler must be less than high bound"
        assert len(uniform) == 2, "Uniform sampler must be defined by two values"

        self.uniform = torch.tensor(uniform)
        self.normal = None

        if noise is not None:
            assert noise[1] > 0, "Standard deviation of normal sampler must be positive"
            assert len(noise) == 2, "Normal sampler must be defined by two values"

            self.normal = torch.tensor(noise)

    def sample(self, shape: torch.Size | Tuple) -> torch.FloatTensor:
        result = torch.zeros(shape).uniform_(self.uniform[0], self.uniform[1], generator=self.generator)

        if self.normal is not None:
            result += torch.zeros(shape).normal_(self.normal[0], self.normal[1], generator=self.generator)
            # Don't allow values to go out of bounds
            result = torch.clip(result, self.uniform[0], self.uniform[0])

        return result

    @property
    def mean(self):
        uniform_mean = (self.uniform[0] + self.uniform[1]) / 2

        return uniform_mean + (self.normal[0] if self.normal is not None else 0)

    @property
    def variance(self):
        uniform_variance = ((self.uniform[0] - self.uniform[1]) ** 2) / 12

        # Variables are independent
        return uniform_variance + (self.normal[1] ** 2 if self.normal is not None else 0)

    @staticmethod
    def from_cli(parameters):
        return Uniform(parameters['uniform'], parameters.get('noise'))
