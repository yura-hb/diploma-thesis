
import torch

from .scheduling_rule import SchedulingRule, MachineState, WaitInfo
from environment.job import Job


class LWRKMODSchedulingRule(SchedulingRule):
    """
    Least Work Remaining + Modified Operational Due date rule. Check implementation of the rules separately.
    """

    def __call__(self, machine_state: MachineState) -> Job | WaitInfo:
        due_at = torch.FloatTensor([job.due_at for job in machine_state.queue])
        processing_times = torch.FloatTensor([
            job.current_operation_processing_time_on_machine for job in machine_state.queue
        ])
        remaining_processing_times = torch.FloatTensor([job.remaining_processing_time() for job in machine_state.queue])
        finish_at = processing_times + machine_state.now

        mod, _ = torch.max(torch.vstack([due_at, finish_at]), dim=0)
        index = torch.argmin(mod + remaining_processing_times)

        return machine_state.queue[index]
