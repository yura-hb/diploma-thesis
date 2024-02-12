from .scheduling_rule import *


class DPTLWKRSchedulingRule(SchedulingRule):
    """
    Double Processing Time + Least Work Remaining
    """

    def __call__(self, machine: Machine, now: float) -> Job | WaitInfo:
        processing_times = torch.FloatTensor(
            [job.current_operation_processing_time_on_machine for job in machine.queue]
        )
        next_remaining_processing_times = torch.FloatTensor(
            [job.next_remaining_processing_time(self.reduction_strategy) for job in machine.queue]
        )

        result = 2 * processing_times + next_remaining_processing_times
        index = torch.argmin(result)

        return machine.queue[index]
