from .scheduling_rule import *


class FIFOSchedulingRule(SchedulingRule):
    @property
    def selector(self):
        return lambda _: 0

    def criterion(self, machine: Machine, now: float) -> torch.FloatTensor:
        return torch.zeros(len(machine.queue))
