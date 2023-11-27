import torch

from .scheduling_rule import SchedulingRule, MachineState, WaitInfo
from problem.job import Job


class SPMWKSchedulingRule(SchedulingRule):
    """
    Slack pe Remaining Work scheduling rule
    """

    def __call__(self, machine_state: MachineState) -> Job | WaitInfo:
        slack = torch.LongTensor(
            [job.slack_upon_now(machine_state.now) for job in machine_state.queue]
        )
        remaining_processing_times = torch.LongTensor(
            [job.remaining_processing_time for job in machine_state.queue]
        )

        ratio = slack / remaining_processing_times

        index = torch.argmin(ratio)

        return machine_state.queue[index]
