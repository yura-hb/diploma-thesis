
import torch

from .scheduling_rule import SchedulingRule, MachineState, WaitInfo
from environment.job import Job


class LROSchedulingRule(SchedulingRule):
    """
    Least Remaining Operations rule, i.e. selects jobs, which has the smallest number of remaining operations
    """

    def __call__(self, machine_state: MachineState) -> Job | WaitInfo:
        values = torch.FloatTensor([job.remaining_operations_count for job in machine_state.queue])
        idx = torch.argmin(values)

        return machine_state.queue[idx]
