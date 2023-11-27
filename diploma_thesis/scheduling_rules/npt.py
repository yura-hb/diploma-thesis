

import torch

from .scheduling_rule import SchedulingRule, MachineState, WaitInfo
from problem.job import Job


class NPTSchedulingRule(SchedulingRule):
    """
    Next Processing Time rule, i.e. selects jobs, in which the processing time of the next operation in the job is
    the smallest
    """

    def __call__(self, machine_state: MachineState) -> Job | WaitInfo:
        values = [job.next_operation_processing_time for job in machine_state.queue]
        idx = torch.argmin(values)

        return machine_state.queue[idx]
