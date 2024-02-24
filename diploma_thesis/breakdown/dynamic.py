import torch

from environment import Breakdown, Machine
from typing import Dict
from sampler import NumericSampler, numeric_sampler_from_cli
from dataclasses import dataclass


class Dynamic(Breakdown):

    @dataclass
    class Configuration:
        arrival: NumericSampler
        duration: NumericSampler

    def __init__(self, configuration: Configuration):
        super().__init__()

        self.configuration = configuration

    def connect(self, generator: torch.Generator):
        self.configuration.arrival.connect(generator)
        self.configuration.duration.connect(generator)

    def sample_next_breakdown_time(self, machine: Machine):
        return self.configuration.arrival.sample((1,))

    def sample_repair_duration(self, machine: Machine):
        return self.configuration.duration.sample((1,))

    @staticmethod
    def from_cli(parameters: Dict):
        return Dynamic(Dynamic.Configuration(
            arrival=numeric_sampler_from_cli(parameters['breakdown_arrival']),
            duration=numeric_sampler_from_cli(parameters['repair_duration'])
        ))
