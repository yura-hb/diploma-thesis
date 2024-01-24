

import torch

from .scheduling_rule import SchedulingRule, MachineState, WaitInfo
from environment.job import Job


class MSSchedulingRule(SchedulingRule):
    """
    Minimum Slack scheduling rule, i.e. the rule selects jobs with the minimum slack
    """

    def __call__(self, machine_state: MachineState) -> Job | WaitInfo:
        slack = torch.FloatTensor(
            [job.slack_upon_now(machine_state.now) for job in machine_state.queue]
        )

        index = torch.argmin(slack)

        return machine_state.queue[index]
