
import torch

from .scheduling_rule import SchedulingRule, MachineState, WaitInfo
from problem.job import Job


class ATCSchedulingRule(SchedulingRule):
    """
    Apparent Tardiness Cost scheduling rule

    Source: https://www.jstor.org/stable/2632177
    """

    # TODO: - Verify

    def __call__(self, machine_state: MachineState) -> Job | WaitInfo:
        processing_times = torch.LongTensor(
            [job.current_operation_processing_time for job in machine_state.queue]
        )
        due_times = torch.LongTensor(
            [job.time_until_due(machine_state.now) for job in machine_state.queue]
        )

        cost = due_times - machine_state.now - processing_times
        cost = torch.clip(cost, min=0)

        priority = torch.exp(-cost / (0.05 * torch.mean(processing_times)))
        priority /= processing_times

        index = torch.argmax(priority)

        return machine_state.queue[index]
