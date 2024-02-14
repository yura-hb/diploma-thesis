from .scheduling_rule import *


class NPTSchedulingRule(SchedulingRule):
    """
    Next Processing Time rule, i.e. selects jobs, in which the processing time of the next operation in the job is
    the smallest
    """

    @property
    def selector(self):
        return torch.argmin

    def criterion(self, machine: Machine, now: float) -> torch.FloatTensor:
        values = torch.FloatTensor([
            job.next_operation_processing_time(self.reduction_strategy) for job in machine.queue
        ])

        return values
