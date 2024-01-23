

from .scheduling_rule import SchedulingRule, MachineState, WaitInfo
from environment.job import Job


class FIFOSchedulingRule(SchedulingRule):

    def __call__(self, machine_state: MachineState) -> Job | WaitInfo:
        return machine_state.queue[0]
