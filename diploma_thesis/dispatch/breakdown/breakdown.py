from abc import ABCMeta

import torch

import environment
from utils import Loggable


class Breakdown(Loggable, metaclass=ABCMeta):

    def connect(self, generator: torch.Generator):
        pass

    def sample_next_breakdown_time(self, machine: 'environment.Machine') -> float:
        pass

    def sample_repair_duration(self, machine: 'environment.Machine') -> float:
        pass
