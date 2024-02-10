from .scheduling_rule import *

# TODO: Pass Reduction strategy


class AVPROSchedulingRule(SchedulingRule):
    """
    Average Processing Time per Operation scheduling rule
    """

    def __call__(self, machine: Machine, now: float) -> Job | WaitInfo:
        remaining_processing_times = torch.FloatTensor(
            [job.remaining_processing_time() for job in machine.queue]
        )
        # Add one to avoid division on zero
        operation_count = torch.FloatTensor(
            [job.remaining_operations_count for job in machine.queue]
        ) + 1

        ratio = remaining_processing_times / operation_count

        index = torch.argmin(ratio)

        return machine.queue[index]