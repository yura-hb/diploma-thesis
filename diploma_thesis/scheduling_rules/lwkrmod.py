
import torch

from .scheduling_rule import SchedulingRule, MachineState, WaitInfo
from environment.job import Job


class LWRKMODSchedulingRule(SchedulingRule):
    """
    Least Work Remaining + Modified Operational Due date rule. Check implementation of the rules separately.
    """

    # TODO: Verify

    def __call__(self, machine_state: MachineState) -> Job | WaitInfo:
        due_at = torch.LongTensor([job.due_at for job in machine_state.queue])
        processing_times = torch.LongTensor([job.current_operation_processing_time for job in machine_state.queue])
        remaining_processing_times = torch.LongTensor([job.remaining_processing_time for job in machine_state.queue])
        finish_at = processing_times + machine_state.now

        mod = torch.max(torch.vstack([due_at, finish_at]), dim=0)
        index = torch.argmin(mod + processing_times + remaining_processing_times)

        return machine_state.queue[index]
