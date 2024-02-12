from .scheduling_rule import *


class CRSchedulingRule(SchedulingRule):
    """
    Critical Ratio scheduling rule, i.e. the rule selects jobs with the lowest ratio of due time to remaining
    processing time
    """

    def __call__(self, machine: Machine, now: float) -> Job | WaitInfo:
        remaining_processing_times = torch.FloatTensor(
            [job.remaining_processing_time(self.reduction_strategy) for job in machine.queue]
        )
        due_times = torch.FloatTensor(
            [job.time_until_due(now) for job in machine.queue]
        )

        ratio = due_times / remaining_processing_times
        index = torch.argmin(ratio)

        return machine.queue[index]
