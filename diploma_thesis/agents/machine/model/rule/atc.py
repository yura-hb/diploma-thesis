
from .scheduling_rule import *


class ATCSchedulingRule(SchedulingRule):
    """
    Apparent Tardiness Cost scheduling rule

    Source: https://www.jstor.org/stable/2632177
    """

    def __call__(self, machine: Machine, now: float) -> Job | WaitInfo:
        processing_times = torch.FloatTensor(
            [job.current_operation_processing_time_on_machine for job in machine.queue]
        )
        due_times = torch.FloatTensor(
            [job.time_until_due(now) for job in machine.queue]
        )

        slack = due_times - now - processing_times
        slack = torch.clip(slack, min=0)

        priority = torch.exp(-slack / (0.05 * torch.mean(processing_times)))
        priority /= processing_times

        index = torch.argmax(priority)

        return machine.queue[index]
