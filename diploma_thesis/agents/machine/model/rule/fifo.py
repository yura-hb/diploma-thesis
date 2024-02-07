from .scheduling_rule import *


class FIFOSchedulingRule(SchedulingRule):

    def __call__(self, machine: Machine, now: float) -> Job | WaitInfo:
        return machine.queue[0]
