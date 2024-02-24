import torch

from environment import Breakdown, Machine
from typing import Dict


class No(Breakdown):

    def connect(self, generator: torch.Generator):
        pass

    def sample_next_breakdown_time(self, machine: Machine):
        return float('inf')

    def sample_repair_duration(self, machine: Machine):
        return 0

    @staticmethod
    def from_cli(parameters: Dict):
        return No()
