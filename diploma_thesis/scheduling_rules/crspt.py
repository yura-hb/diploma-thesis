
import torch

from .scheduling_rule import SchedulingRule, MachineState, WaitInfo
from environment.job import Job


class CRSPTSchedulingRule(SchedulingRule):
    """
    Critical Ratio + Shortest Processing Time scheduling rule, i.e.
    the rule selects jobs with the lowest ratio of due time to remaining processing time and current operation time
    """

    def __call__(self, machine_state: MachineState) -> Job | WaitInfo:
        processing_times = torch.LongTensor(
            [job.current_operation_processing_time for job in machine_state.queue]
        )
        remaining_processing_times = torch.LongTensor(
            [job.remaining_processing_time for job in machine_state.queue]
        )
        due_times = torch.LongTensor(
            [job.time_until_due(machine_state.now) for job in machine_state.queue]
        )

        ratio = due_times / (remaining_processing_times + processing_times)
        index = torch.argmin(ratio)

        return machine_state.queue[index]
