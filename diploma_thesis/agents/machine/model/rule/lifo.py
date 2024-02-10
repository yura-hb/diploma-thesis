from .scheduling_rule import *


class LIFOSchedulingRule(SchedulingRule):

    def __call__(self, machine: Machine, now: float) -> Job | WaitInfo:
        return machine.queue[-1]
