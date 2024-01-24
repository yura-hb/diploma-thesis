
import torch

from .scheduling_rule import SchedulingRule, MachineState, WaitInfo
from environment.job import Job


class ATCSchedulingRule(SchedulingRule):
    """
    Apparent Tardiness Cost scheduling rule

    Source: https://www.jstor.org/stable/2632177
    """

    def __call__(self, machine_state: MachineState) -> Job | WaitInfo:
        processing_times = torch.FloatTensor(
            [job.current_operation_processing_time_on_machine for job in machine_state.queue]
        )
        due_times = torch.FloatTensor(
            [job.time_until_due(machine_state.now) for job in machine_state.queue]
        )

        cost = due_times - machine_state.now - processing_times
        cost = torch.clip(cost, min=0)

        priority = torch.exp(-cost / (0.05 * torch.mean(processing_times)))
        priority /= processing_times

        index = torch.argmax(priority)

        return machine_state.queue[index]
