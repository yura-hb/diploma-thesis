from .scheduling_rule import *


class EDDSchedulingRule(SchedulingRule):
    """
    Earliest Due Date rule, i.e. selects jobs, in which due at is the shortest
    """

    def __call__(self, machine: Machine, now: float) -> Job | WaitInfo:
        values = torch.FloatTensor([job.due_at for job in machine.queue])
        idx = torch.argmin(values)

        return machine.queue[idx]
