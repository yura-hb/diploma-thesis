
import torch

from .scheduling_rule import SchedulingRule, MachineState, WaitInfo
from environment.job import Job


class SPTSchedulingRule(SchedulingRule):
    """
    Shortest Processing Time rule, i.e. selects jobs, in which current operation has the smallest operation time
    """

    def __call__(self, machine_state: MachineState) -> Job | WaitInfo:
        values = [job.current_operation_processing_time_on_machine for job in machine_state.queue]
        idx = torch.argmin(values)

        return machine_state.queue[idx]
