from .scheduling_rule import *


class MODSchedulingRule(SchedulingRule):
    """
    Modified operational due date
    """

    def selector(self):
        return torch.argmin

    def criterion(self, machine: Machine, now: float) -> torch.FloatTensor:
        processing_times = torch.FloatTensor([
            job.current_operation_processing_time_on_machine for job in machine.queue
        ])
        due_times = torch.FloatTensor([job.due_at for job in machine.queue])

        operation_completed_at = processing_times + now
        stacked = torch.vstack([due_times, operation_completed_at])

        mod, _ = torch.max(stacked, dim=0)

        return mod
