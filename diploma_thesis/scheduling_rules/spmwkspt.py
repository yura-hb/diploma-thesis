
import torch

from .scheduling_rule import SchedulingRule, MachineState, WaitInfo
from environment.job import Job


class SPMWKSPTSchedulingRule(SchedulingRule):
    """
    Slack per Remaining Work + Shortest Processing Time scheduling rule
    """
    # TODO: - Verify

    def __call__(self, machine_state: MachineState) -> Job | WaitInfo:
        slack = torch.FloatTensor(
            [job.slack_upon_now(machine_state.now) for job in machine_state.queue]
        )
        operation_processing_times = torch.FloatTensor(
            [job.current_operation_processing_time_on_machine for job in machine_state.queue]
        )
        remaining_processing_times = torch.FloatTensor(
            [job.remaining_processing_time() for job in machine_state.queue]
        )

        ratio = (slack / remaining_processing_times) + operation_processing_times

        index = torch.argmin(ratio)

        return machine_state.queue[index]
