from .scheduling_rule import *


class LWRKMODSchedulingRule(SchedulingRule):
    """
    Least Work Remaining + Modified Operational Due date rule. Check implementation of the rules separately.
    """

    def selector(self):
        return torch.argmin

    def criterion(self, machine: Machine, now: float) -> torch.FloatTensor:
        due_at = torch.FloatTensor([job.due_at for job in machine.queue])
        processing_times = torch.FloatTensor([
            job.current_operation_processing_time_on_machine for job in machine.queue
        ])
        remaining_processing_times = torch.FloatTensor([
            job.remaining_processing_time(self.reduction_strategy) for job in machine.queue
        ])
        finish_at = processing_times + now

        mod, _ = torch.max(torch.vstack([due_at, finish_at]), dim=0)
        values = mod + remaining_processing_times

        return values
