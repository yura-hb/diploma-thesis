
import torch

from environment import Job, SchedulingRule, Machine, WaitInfo


class SPMWKSPTSchedulingRule(SchedulingRule):
    """
    Slack per Remaining Work + Shortest Processing Time scheduling rule
    """
    # TODO: - Verify

    def __call__(self, machine: Machine, now: float) -> Job | WaitInfo:
        slack = torch.FloatTensor(
            [job.slack_upon_moment(now) for job in machine.queue]
        )
        operation_processing_times = torch.FloatTensor(
            [job.current_operation_processing_time_on_machine for job in machine.queue]
        )
        remaining_processing_times = torch.FloatTensor(
            [job.remaining_processing_time() for job in machine.queue]
        )

        ratio = (slack / remaining_processing_times) + operation_processing_times

        index = torch.argmin(ratio)

        return machine.queue[index]
