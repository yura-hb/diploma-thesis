
from typing import Dict

import simpy

from environment import Configuration, JobSampler
from .dynamic import (CLI as DynamicJobSamplerFromCLI,
                      Builder as DynamicJobSamplerBuilder,
                      Sampler as DynamicJobSampler)
from .static import (Sampler as StaticJobSampler)

key_to_class = {
    "dynamic": DynamicJobSamplerFromCLI
}


def from_cli(problem: Configuration, environment: simpy.Environment, configuration: Dict) -> 'JobSampler':
    cls = key_to_class[configuration['kind']]

    return cls.from_cli(problem, environment, configuration['parameters'])
