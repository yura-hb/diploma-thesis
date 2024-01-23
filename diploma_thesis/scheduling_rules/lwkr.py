
import torch

from .scheduling_rule import SchedulingRule, MachineState, WaitInfo
from environment.job import Job


class LWRKSchedulingRule(SchedulingRule):
    """
    Least Work Remaining rule, i.e. selects jobs, in which the remaining time of the job is the smallest.
    """

    def __call__(self, machine_state: MachineState) -> Job | WaitInfo:
        values = [job.remaining_processing_time for job in machine_state.queue]
        idx = torch.argmin(values)

        return machine_state.queue[idx]
