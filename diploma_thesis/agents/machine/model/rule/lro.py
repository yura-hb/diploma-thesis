from .scheduling_rule import *


class LROSchedulingRule(SchedulingRule):
    """
    Least Remaining Operations rule, i.e. selects jobs, which has the smallest number of remaining operations
    """

    def selector(self):
        return torch.argmin

    def criterion(self, machine: Machine, now: float) -> torch.FloatTensor:
        values = torch.FloatTensor([job.remaining_operations_count for job in machine.queue])

        return values
