from .scheduling_rule import *


class SPMWKSPTSchedulingRule(SchedulingRule):
    """
    Slack per Remaining Work + Shortest Processing Time scheduling rule
    """

    def selector(self):
        return torch.argmin

    def criterion(self, machine: Machine, now: float) -> torch.FloatTensor:
        slack = torch.FloatTensor(
            [job.slack_upon_moment(now, self.reduction_strategy) for job in machine.queue]
        )
        operation_processing_times = torch.FloatTensor(
            [job.current_operation_processing_time_on_machine for job in machine.queue]
        )
        remaining_processing_times = torch.FloatTensor(
            [job.remaining_processing_time(self.reduction_strategy) for job in machine.queue]
        )

        ratio = (slack / remaining_processing_times) + operation_processing_times

        return ratio
