from .scheduling_rule import *


class MDDSchedulingRule(SchedulingRule):
    """
    Modified due date selection rule, i.e.
        selects job with the smallest value of max(due_at, operation_completed_at)
    """

    @property
    def selector(self):
        return torch.argmin

    def criterion(self, machine: Machine, now: float) -> torch.FloatTensor:
        processing_times = torch.FloatTensor(
            [job.next_remaining_processing_time(self.reduction_strategy) for job in machine.queue]
        )
        due_times = torch.FloatTensor([job.due_at for job in machine.queue])

        finish_at = processing_times + now

        stacked = torch.vstack([due_times, finish_at])
        mod, _ = torch.max(stacked, dim=0)

        return mod
