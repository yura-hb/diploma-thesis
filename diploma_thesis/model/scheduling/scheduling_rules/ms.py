

import torch

import environment
from model.scheduling.scheduling_rule import SchedulingRule, WaitInfo


class MSSchedulingRule(SchedulingRule):
    """
    Minimum Slack scheduling rule, i.e. the rule selects jobs with the minimum slack
    """

    def __call__(self, machine: environment.Machine, now: float) -> environment.Job | WaitInfo:
        slack = torch.FloatTensor(
            [job.slack_upon_moment(now) for job in machine.queue]
        )

        index = torch.argmin(slack)

        return machine.queue[index]
