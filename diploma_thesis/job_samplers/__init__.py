
from typing import Dict

import simpy

from environment import Configuration, JobSampler
from .dynamic import (CLI as DynamicJobSamplerFromCLI,
                      Builder as DynamicJobSamplerBuilder,
                      Sampler as DynamicJobSampler)
from .static import (Sampler as StaticJobSampler)

key_to_sampler_builder = {
    "dynamic": DynamicJobSamplerFromCLI
}


def from_cli_arguments(problem: Configuration, environment: simpy.Environment, configuration: Dict) -> 'JobSampler':
    sampler = key_to_sampler_builder[configuration['id']]

    return sampler.from_cli_arguments(problem, environment, configuration['parameters'])
