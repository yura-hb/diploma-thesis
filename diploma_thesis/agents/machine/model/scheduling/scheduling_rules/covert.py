
import torch

from environment import SchedulingRule, WaitInfo, Machine, Job


# TODO: Pass Reduction strategy

class COVERTSchedulingRule(SchedulingRule):
    """
    Cost Over Time scheduling rule, i.e. selects jobs with maximium cost.
    We assume that the cost is
    """

    def __call__(self, machine: Machine, now: float) -> Job | WaitInfo:
        processing_times = torch.FloatTensor([job.remaining_processing_time() for job in machine.queue])

        available_slack = torch.FloatTensor([job.slack_upon_moment(now) for job in machine.queue])
        available_slack = torch.clip(available_slack, 0, None)

        priority = (1 - available_slack / (0.05 * torch.mean(processing_times)))
        priority = torch.clip(priority, 0, None)
        priority /= processing_times

        index = torch.argmax(priority)

        return machine.queue[index]
