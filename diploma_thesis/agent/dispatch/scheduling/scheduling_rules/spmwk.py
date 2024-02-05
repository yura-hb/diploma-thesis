import torch

from environment import Job, SchedulingRule, Machine, WaitInfo


class SPMWKSchedulingRule(SchedulingRule):
    """
    Slack pe Remaining Work scheduling rule
    """

    def __call__(self, machine: Machine, now: float) -> Job | WaitInfo:
        slack = torch.FloatTensor(
            [job.slack_upon_moment(now) for job in machine.queue]
        )
        remaining_processing_times = torch.FloatTensor(
            [job.remaining_processing_time() for job in machine.queue]
        )

        ratio = slack / remaining_processing_times

        index = torch.argmin(ratio)

        return machine.queue[index]
