
import torch

import environment
from model.scheduling.scheduling_rule import SchedulingRule, WaitInfo


class MONSchedulingRule(SchedulingRule):
    """
    Montagne's heuristics,
        i.e. SPT + additional slack factor
    """

    def __call__(self, machine: environment.Machine, now: float) -> environment.Job | WaitInfo:
        processing_times = torch.FloatTensor([
            job.current_operation_processing_time_on_machine for job in machine.queue
        ])
        due_times = torch.FloatTensor([job.due_at for job in machine.queue])

        ratio = due_times / torch.sum(processing_times)
        priority = ratio / processing_times

        index = torch.argmax(priority)

        return machine.queue[index]
