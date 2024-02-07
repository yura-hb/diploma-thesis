from .scheduling_rule import *


class LROSchedulingRule(SchedulingRule):
    """
    Least Remaining Operations rule, i.e. selects jobs, which has the smallest number of remaining operations
    """

    def __call__(self, machine: Machine, now: float) -> Job | WaitInfo:
        values = torch.FloatTensor([job.remaining_operations_count for job in machine.queue])
        idx = torch.argmin(values)

        return machine.queue[idx]
