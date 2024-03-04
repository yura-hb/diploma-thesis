from .scheduling_rule import *


class SWTSchedulingRule(SchedulingRule):
    """
    Shortest Waiting  rule, i.e. selects jobs, in which current operation has the smallest waiting time
    """

    @property
    def selector(self):
        return torch.argmin

    def criterion(self, machine: Machine, now: float) -> torch.FloatTensor:
        values = torch.FloatTensor([
            job.current_operation_waiting_time_on_machine(now) for job in machine.queue
        ])

        return values
