
import torch

import environment
from model.scheduling.scheduling_rule import SchedulingRule, WaitInfo


class EDDSchedulingRule(SchedulingRule):
    """
    Earliest Due Date rule, i.e. selects jobs, in which due at is the shortest
    """

    def __call__(self, machine: environment.Machine, now: float) -> environment.Job | WaitInfo:
        values = torch.FloatTensor([job.due_at for job in machine.queue])
        idx = torch.argmin(values)

        return machine.queue[idx]
