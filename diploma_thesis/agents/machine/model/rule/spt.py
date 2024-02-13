from .scheduling_rule import *


class SPTSchedulingRule(SchedulingRule):
    """
    Shortest Processing Time rule, i.e. selects jobs, in which current operation has the smallest operation time
    """

    def selector(self):
        return torch.argmin

    def criterion(self, machine: Machine, now: float) -> torch.FloatTensor:
        values = torch.FloatTensor([
            job.current_operation_processing_time_on_machine for job in machine.queue
        ])

        return values
