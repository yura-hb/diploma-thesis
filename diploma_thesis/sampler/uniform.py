
import torch

from typing import Tuple
from .numeric_sampler import NumericSampler


class Uniform(NumericSampler):
    def __init__(self, uniform: Tuple, noise: Tuple):
        assert uniform[0] < uniform[1], "Low bound of uniform sampler must be less than high bound"
        assert len(uniform) == 2, "Uniform sampler must be defined by two values"

        self.uniform = torch.distributions.Uniform(low=uniform[0], high=uniform[1])
        self.normal = None

        if noise is not None:
            assert noise[1] > 0, "Standard deviation of normal sampler must be positive"
            assert len(noise) == 2, "Normal sampler must be defined by two values"

            self.normal = torch.distributions.Normal(loc=noise[0], scale=noise[1])

    def sample(self, shape: torch.Size | Tuple) -> torch.FloatTensor:
        result = self.uniform.sample(shape)

        if self.normal is not None:
            result += self.normal.sample(shape)
            # Don't allow values to go out of bounds
            result = torch.clip(result, self.uniform.low, self.uniform.high)

        return result

    @property
    def mean(self):
        return self.uniform.mean + (self.normal.mean if self.normal is not None else 0)

    @property
    def variance(self):
        # Variables are independent
        return self.uniform.variance + (self.normal.variance if self.normal is not None else 0)

    @staticmethod
    def from_cli(parameters):
        return Uniform(parameters['uniform'], parameters.get('noise'))
