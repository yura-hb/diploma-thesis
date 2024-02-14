from .scheduling_rule import *


class CRSPTSchedulingRule(SchedulingRule):
    """
    Critical Ratio + Shortest Processing Time scheduling rule, i.e.
    the rule selects jobs with the lowest ratio of due time to remaining processing time and current operation time
    """

    @property
    def selector(self):
        return torch.argmin

    def criterion(self, machine: 'Machine', now: float) -> torch.FloatTensor:
        remaining_processing_times = torch.FloatTensor(
            [job.remaining_processing_time(self.reduction_strategy) for job in machine.queue]
        )
        due_times = torch.FloatTensor(
            [job.time_until_due(now) for job in machine.queue]
        )
        processing_times = torch.FloatTensor(
            [job.current_operation_processing_time_on_machine for job in machine.queue]
        )

        ratio = due_times / remaining_processing_times + processing_times

        return ratio
