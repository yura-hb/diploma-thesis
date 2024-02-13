from .scheduling_rule import *


class LPTSchedulingRule(SchedulingRule):
    """
    Longest Processing Time rule, i.e. selects jobs, in which current operation has the largest operation time
    """

    def selector(self):
        return torch.argmax

    def criterion(self, machine: Machine, now: float) -> torch.FloatTensor:
        values = torch.FloatTensor([job.current_operation_processing_time_on_machine for job in machine.queue])

        return values
