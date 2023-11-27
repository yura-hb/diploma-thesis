
import torch

from .scheduling_rule import SchedulingRule, MachineState, WaitInfo
from problem.job import Job


class LWRKSPTSchedulingRule(SchedulingRule):
    """
    Least Work Remaining + Shortest Processing Time rule,
        i.e. selects jobs, in which satisfy both criteria (for reference check lwrk.py and spt.py)
    """

    def __call__(self, machine_state: MachineState) -> Job | WaitInfo:
        values = [
            job.remaining_processing_time + job.current_operation_processing_time for job in machine_state.queue
        ]
        idx = torch.argmin(values)

        return machine_state.queue[idx]
