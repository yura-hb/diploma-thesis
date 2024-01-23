

import torch

from .scheduling_rule import SchedulingRule, MachineState, WaitInfo
from environment.job import Job


class COVERTSchedulingRule(SchedulingRule):
    """
    Cost Over Time scheduling rule, i.e. selects jobs with maximium cost.
    We assume that the cost is
    """

    # TODO: - Verify

    def __call__(self, machine_state: MachineState) -> Job | WaitInfo:
        processing_times = torch.LongTensor([job.remaining_processing_time for job in machine_state.queue])
        dues = torch.LongTensor([job.due_at for job in machine_state.queue])

        available_slack = (dues - processing_times - machine_state.now)
        available_slack = torch.clip(available_slack, 0, None)

        priority = (1 - available_slack / (0.05 * torch.mean(processing_times)))
        priority = torch.clip(priority, 0, None)
        priority /= processing_times

        index = torch.argmax(priority)

        return machine_state.queue[index]
