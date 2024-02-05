import torch

import environment
from model.scheduling.scheduling_rule import SchedulingRule, WaitInfo


class SPMWKSchedulingRule(SchedulingRule):
    """
    Slack pe Remaining Work scheduling rule
    """

    def __call__(self, machine: environment.Machine, now: float) -> environment.Job | WaitInfo:
        slack = torch.FloatTensor(
            [job.slack_upon_moment(now) for job in machine.queue]
        )
        remaining_processing_times = torch.FloatTensor(
            [job.remaining_processing_time() for job in machine.queue]
        )

        ratio = slack / remaining_processing_times

        index = torch.argmin(ratio)

        return machine.queue[index]
