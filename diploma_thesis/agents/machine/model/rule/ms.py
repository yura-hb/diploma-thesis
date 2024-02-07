from .scheduling_rule import *


class MSSchedulingRule(SchedulingRule):
    """
    Minimum Slack scheduling rule, i.e. the rule selects jobs with the minimum slack
    """

    def __call__(self, machine: Machine, now: float) -> Job | WaitInfo:
        slack = torch.FloatTensor(
            [job.slack_upon_moment(now) for job in machine.queue]
        )

        index = torch.argmin(slack)

        return machine.queue[index]
