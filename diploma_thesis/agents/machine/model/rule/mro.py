from .scheduling_rule import *


class MROSchedulingRule(SchedulingRule):
    """
    Most Remaining Operations rule, i.e. selects jobs, which has the largest number of remaining operations
    """

    @property
    def selector(self):
        return torch.argmax

    def criterion(self, machine: Machine, now: float) -> torch.FloatTensor:
        values = torch.FloatTensor([job.remaining_operations_count for job in machine.queue])

        return values
