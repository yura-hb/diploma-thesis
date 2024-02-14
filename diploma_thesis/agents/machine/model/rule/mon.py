from .scheduling_rule import *


class MONSchedulingRule(SchedulingRule):
    """
    Montagne's heuristics,
        i.e. SPT + additional slack factor
    """

    @property
    def selector(self):
        return torch.argmax

    def criterion(self, machine: Machine, now: float) -> torch.FloatTensor:
        processing_times = torch.FloatTensor([
            job.current_operation_processing_time_on_machine for job in machine.queue
        ])
        due_times = torch.FloatTensor([job.due_at for job in machine.queue])

        ratio = due_times / torch.sum(processing_times)
        priority = ratio / processing_times

        return priority
