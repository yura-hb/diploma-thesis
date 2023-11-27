import torch

from .scheduling_rule import SchedulingRule, MachineState, WaitInfo
from problem.job import Job


class AVPROSchedulingRule(SchedulingRule):
    """
    Average Processing Time per Operation scheduling rule
    """

    def __call__(self, machine_state: MachineState) -> Job | WaitInfo:
        remaining_processing_times = torch.LongTensor(
            [job.remaining_processing_time for job in machine_state.queue]
        )
        # Add one to avoid division on zero
        operation_count = torch.LongTensor(
            [job.remaining_operations_count for job in machine_state.queue]
        ) + 1

        ratio = remaining_processing_times / operation_count

        index = torch.argmax(ratio)

        return machine_state.queue[index]
