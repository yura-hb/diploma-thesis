
from .sampler import Sampler
from .numeric_sampler import NumericSampler
from .constant import Constant
from .uniform import Uniform
from .permutation import Permutation
from .exponential import Exponential

from functools import partial
from utils import from_cli

key_to_cls = {
    'constant': Constant,
    'uniform': Uniform,
    'exponential': Exponential
}

numeric_sampler_from_cli = partial(from_cli, key_to_class=key_to_cls)


def permutation_sampler_from_cli(parameters) -> Permutation:
    return Permutation.from_cli(parameters)
