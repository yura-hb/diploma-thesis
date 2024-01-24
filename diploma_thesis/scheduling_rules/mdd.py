
import torch

from .scheduling_rule import SchedulingRule, MachineState, WaitInfo
from environment.job import Job


class MDDSchedulingRule(SchedulingRule):
    """
    Modified due date selection rule, i.e.
        selects job with the smallest value of max(due_at, operation_completed_at)
    """

    def __call__(self, machine_state: MachineState) -> Job | WaitInfo:
        processing_times = torch.FloatTensor(
            [job.current_operation_processing_time_on_machine for job in machine_state.queue]
        )
        due_times = torch.FloatTensor([job.due_at for job in machine_state.queue])

        finish_at = processing_times + machine_state.now

        stacked = torch.vstack([due_times, finish_at])
        mod, _ = torch.max(stacked, dim=1)
        index = torch.argmin(mod)

        return machine_state.queue[index]
