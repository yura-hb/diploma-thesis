
import torch

from .scheduling_rule import SchedulingRule, MachineState, WaitInfo
from problem.job import Job


class MODSchedulingRule(SchedulingRule):
    """
    Modified operational due date
    """

    def __call__(self, machine_state: MachineState) -> Job | WaitInfo:
        processing_times = [job.current_operation_processing_time for job in machine_state.queue]
        due_times = [job.due_at for job in machine_state.queue]

        operation_completed_at = processing_times + machine_state.now
        stacked = torch.vstack([due_times, operation_completed_at])

        mod = torch.max(stacked, dim=1)
        index = torch.argmin(mod)

        return machine_state.queue[index]