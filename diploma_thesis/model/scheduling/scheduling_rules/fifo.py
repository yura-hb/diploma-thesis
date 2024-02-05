
import environment

from model.scheduling.scheduling_rule import SchedulingRule, WaitInfo


class FIFOSchedulingRule(SchedulingRule):

    def __call__(self, machine: environment.Machine, now: float) -> environment.Job | WaitInfo:
        return machine.queue[0]
