from .scheduling_rule import *


class COVERTSchedulingRule(SchedulingRule):
    """
    Cost Over Time scheduling rule, i.e. selects jobs with maximium cost.
    We assume that the cost is
    """

    def selector(self):
        return torch.argmax

    def criterion(self, machine: Machine, now: float) -> torch.FloatTensor:
        processing_times = torch.FloatTensor([
            job.current_operation_processing_time_on_machine for job in machine.queue
        ])
        available_slack = torch.FloatTensor([
            job.slack_upon_moment(now, self.reduction_strategy) for job in machine.queue
        ])
        available_slack = torch.clip(available_slack, 0, None)

        priority = (1 - available_slack / (0.05 * torch.mean(processing_times)))
        priority = torch.clip(priority, 0, None)
        priority /= processing_times

        return priority