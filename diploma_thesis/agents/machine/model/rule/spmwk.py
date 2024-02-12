from .scheduling_rule import *


class SPMWKSchedulingRule(SchedulingRule):
    """
    Slack per Remaining Work scheduling rule
    """

    def __call__(self, machine: Machine, now: float) -> Job | WaitInfo:
        slack = torch.FloatTensor(
            [job.slack_upon_moment(now, self.reduction_strategy) for job in machine.queue]
        )
        remaining_processing_times = torch.FloatTensor(
            [job.remaining_processing_time(self.reduction_strategy) for job in machine.queue]
        )

        ratio = slack / remaining_processing_times

        index = torch.argmin(ratio)

        return machine.queue[index]
