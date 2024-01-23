
import torch

from .scheduling_rule import SchedulingRule, MachineState, WaitInfo
from environment.job import Job


class MDDSchedulingRule(SchedulingRule):
    """
    Modified due date selection rule, i.e.
        selects job with the smallest value of max(due_at, operation_completed_at)
    """

    def __call__(self, machine_state: MachineState) -> Job | WaitInfo:
        remaining_processing_times = torch.LongTensor(
            [job.current_operation_processing_time for job in machine_state.queue]
        )
        due_times = torch.LongTensor([job.due_at for job in machine_state.queue])

        finish_at = remaining_processing_times + machine_state.now

        stacked = torch.vstack([due_times, finish_at])
        mod = torch.max(stacked, dim=1)
        index = torch.argmin(mod)

        return machine_state.queue[index]
