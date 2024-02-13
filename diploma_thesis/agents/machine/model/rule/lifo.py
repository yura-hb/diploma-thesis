from .scheduling_rule import *


class LIFOSchedulingRule(SchedulingRule):

    def selector(self):
        return lambda _: -1

    def criterion(self, machine: Machine, now: float) -> torch.FloatTensor:
        return torch.zeros(len(machine.queue))
