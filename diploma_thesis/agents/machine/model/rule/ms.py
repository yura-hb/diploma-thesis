from .scheduling_rule import *


class MSSchedulingRule(SchedulingRule):
    """
    Minimum Slack scheduling rule, i.e. the rule selects jobs with the minimum slack
    """

    def selector(self):
        return torch.argmin

    def criterion(self, machine: Machine, now: float) -> torch.FloatTensor:
        slack = torch.FloatTensor(
            [job.slack_upon_moment(now, self.reduction_strategy) for job in machine.queue]
        )

        return slack
