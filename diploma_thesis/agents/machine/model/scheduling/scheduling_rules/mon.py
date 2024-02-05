
import torch

from environment import Job, SchedulingRule, Machine, WaitInfo


class MONSchedulingRule(SchedulingRule):
    """
    Montagne's heuristics,
        i.e. SPT + additional slack factor
    """

    def __call__(self, machine: Machine, now: float) -> Job | WaitInfo:
        processing_times = torch.FloatTensor([
            job.current_operation_processing_time_on_machine for job in machine.queue
        ])
        due_times = torch.FloatTensor([job.due_at for job in machine.queue])

        ratio = due_times / torch.sum(processing_times)
        priority = ratio / processing_times

        index = torch.argmax(priority)

        return machine.queue[index]
