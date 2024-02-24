import torch
from abc import ABCMeta
import environment


class Breakdown(metaclass=ABCMeta):

    def connect(self, generator: torch.Generator):
        pass

    def sample_next_breakdown_time(self, machine: 'environment.Machine') -> float:
        pass

    def sample_repair_duration(self, machine: 'environment.Machine') -> float:
        pass
