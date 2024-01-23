
import torch

from .scheduling_rule import SchedulingRule, MachineState, WaitInfo
from environment.job import Job


class EDDSchedulingRule(SchedulingRule):
    """
    Earliest Due Date rule, i.e. selects jobs, in which due at is the shortest
    """

    def __call__(self, machine_state: MachineState) -> Job | WaitInfo:
        values = [job.due_at for job in machine_state.queue]
        idx = torch.argmin(values)

        return machine_state.queue[idx]
