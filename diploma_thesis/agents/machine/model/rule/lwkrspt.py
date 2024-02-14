from .scheduling_rule import *


class LWRKSPTSchedulingRule(SchedulingRule):
    """
    Least Work Remaining + Shortest Processing Time rule,
        i.e. selects jobs, in which satisfy both criteria (for reference check lwrk.py and spt.py)
    """

    @property
    def selector(self):
        return torch.argmin

    def criterion(self, machine: Machine, now: float) -> torch.FloatTensor:
        processing_times = torch.FloatTensor([
            job.current_operation_processing_time_on_machine for job in machine.queue
        ])
        next_remaining_processing_time = torch.FloatTensor([
            job.next_remaining_processing_time(self.reduction_strategy) for job in machine.queue
        ])
        values = 2 * processing_times + next_remaining_processing_time

        return values