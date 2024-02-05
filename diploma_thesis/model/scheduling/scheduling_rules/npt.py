

import torch

import environment
from model.scheduling.scheduling_rule import SchedulingRule, WaitInfo


class NPTSchedulingRule(SchedulingRule):
    """
    Next Processing Time rule, i.e. selects jobs, in which the processing time of the next operation in the job is
    the smallest
    """

    def __call__(self, machine: environment.Machine, now: float) -> environment.Job | WaitInfo:
        values = torch.FloatTensor([job.next_operation_processing_time() for job in machine.queue])
        idx = torch.argmin(values)

        return machine.queue[idx]
