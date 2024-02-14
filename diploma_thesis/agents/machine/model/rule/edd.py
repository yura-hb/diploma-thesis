from .scheduling_rule import *


class EDDSchedulingRule(SchedulingRule):
    """
    Earliest Due Date rule, i.e. selects jobs, in which due at is the shortest
    """

    @property
    def selector(self):
        return torch.argmin

    def criterion(self, machine: Machine, now: float) -> torch.FloatTensor:
        values = torch.FloatTensor([job.due_at for job in machine.queue])

        return values
