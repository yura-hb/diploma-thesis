
import torch

import environment
from model.scheduling.scheduling_rule import SchedulingRule, WaitInfo


class MODSchedulingRule(SchedulingRule):
    """
    Modified operational due date
    """

    def __call__(self, machine: environment.Machine, now: float) -> environment.Job | WaitInfo:
        processing_times = torch.FloatTensor([
            job.current_operation_processing_time_on_machine for job in machine.queue
        ])
        due_times = torch.FloatTensor([job.due_at for job in machine.queue])

        operation_completed_at = processing_times + now
        stacked = torch.vstack([due_times, operation_completed_at])

        mod, _ = torch.max(stacked, dim=1)
        index = torch.argmin(mod)

        return machine.queue[index]
