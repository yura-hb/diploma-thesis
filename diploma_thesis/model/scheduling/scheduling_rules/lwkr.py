
import torch

from environment import Job, SchedulingRule, Machine, WaitInfo


class LWRKSchedulingRule(SchedulingRule):
    """
    Least Work Remaining rule, i.e. selects jobs, in which the remaining time of the job is the smallest.
    """

    def __call__(self, machine: Machine, now: float) -> Job | WaitInfo:
        values = torch.FloatTensor([job.remaining_processing_time() for job in machine.queue])
        idx = torch.argmin(values)

        return machine.queue[idx]
