import torch

from .scheduling_rule import *


class RandomSchedulingRule(SchedulingRule):
    @property
    def selector(self):
        return lambda x: torch.randint(0, len(x), (1,))

    def criterion(self, machine: Machine, now: float) -> torch.FloatTensor:
        return torch.zeros(len(machine.queue))
