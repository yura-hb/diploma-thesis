
from .sampler import Sampler
from .numeric_sampler import NumericSampler
from .constant import Constant
from .uniform import Uniform
from .permutation import Permutation
from .exponential import Exponential

key_to_cls = {
    'constant': Constant,
    'uniform': Uniform,
    'exponential': Exponential
}


def numeric_sampler_from_cli(parameters) -> NumericSampler:
    cls = key_to_cls[parameters['kind']]

    return cls.from_cli(parameters['parameters'])


def permutation_sampler_from_cli(parameters) -> Permutation:
    return Permutation.from_cli(parameters)
