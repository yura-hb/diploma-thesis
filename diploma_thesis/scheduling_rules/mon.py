
import torch

from .scheduling_rule import SchedulingRule, MachineState, WaitInfo
from environment.job import Job


class MONSchedulingRule(SchedulingRule):
    """
    Montagne's heuristics,
        i.e. SPT + additional slack factor
    """

    def __call__(self, machine_state: MachineState) -> Job | WaitInfo:
        processing_times = torch.FloatTensor([
            job.current_operation_processing_time_on_machine for job in machine_state.queue
        ])
        due_times = torch.FloatTensor([job.due_at for job in machine_state.queue])

        ratio = due_times / torch.sum(processing_times)
        priority = ratio / processing_times

        index = torch.argmax(priority)

        return machine_state.queue[index]
