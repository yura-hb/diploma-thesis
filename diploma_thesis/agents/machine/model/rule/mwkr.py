from .scheduling_rule import *


class MWRKSchedulingRule(SchedulingRule):
    """
    Most Work Remaining rule, i.e. selects jobs, in which the remaining time of the job is the largest.
    """

    @property
    def selector(self):
        return torch.argmax

    def criterion(self, machine: Machine, now: float) -> torch.FloatTensor:
        values = torch.FloatTensor([job.remaining_processing_time(self.reduction_strategy) for job in machine.queue])

        return values