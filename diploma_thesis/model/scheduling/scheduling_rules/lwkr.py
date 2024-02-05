
import torch

import environment
from model.scheduling.scheduling_rule import SchedulingRule, WaitInfo


class LWRKSchedulingRule(SchedulingRule):
    """
    Least Work Remaining rule, i.e. selects jobs, in which the remaining time of the job is the smallest.
    """

    def __call__(self, machine: environment.Machine, now: float) -> environment.Job | WaitInfo:
        values = torch.FloatTensor([job.remaining_processing_time() for job in machine.queue])
        idx = torch.argmin(values)

        return machine.queue[idx]
