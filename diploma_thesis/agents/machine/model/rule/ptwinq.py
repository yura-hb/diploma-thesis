from .scheduling_rule import *


class PTWINQSchedulingRule(SchedulingRule):
    """
    Processing Time + Work In Next Queue
    """

    @property
    def selector(self):
        return torch.argmin

    def criterion(self, machine: Machine, now: float) -> torch.FloatTensor:
        processing_time = torch.FloatTensor([
            job.current_operation_processing_time_on_machine for job in machine.queue
        ])
        winq = torch.FloatTensor([
            machine.shop_floor.work_in_next_queue(job) for job in machine.queue
        ])

        return winq + processing_time
