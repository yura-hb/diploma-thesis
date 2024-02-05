
import torch

import environment
from model.scheduling.scheduling_rule import SchedulingRule, WaitInfo


class SPTSchedulingRule(SchedulingRule):
    """
    Shortest Processing Time rule, i.e. selects jobs, in which current operation has the smallest operation time
    """

    def __call__(self, machine: environment.Machine, now: float) -> environment.Job | WaitInfo:
        values = [job.current_operation_processing_time_on_machine for job in machine.queue]
        idx = torch.argmin(values)

        return machine.queue[idx]
