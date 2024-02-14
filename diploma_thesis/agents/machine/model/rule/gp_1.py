from .scheduling_rule import *


class GP1SchedulingRule(SchedulingRule):
    """
    Genetic Programming 1 scheduling rule. Taken from external/PhD-Thesis-Projects/FJSP/sequencing.py
    """

    @property
    def selector(self):
        return torch.argmin

    def criterion(self, machine: Machine, now: float) -> torch.FloatTensor:
        operation_processing_time = torch.FloatTensor(
            [job.current_operation_processing_time_on_machine for job in machine.queue]
        )
        next_remaining_processing_time = torch.FloatTensor(
            [job.next_remaining_processing_time(self.reduction_strategy) for job in machine.queue]
        )
        winq = torch.FloatTensor([
            machine.shop_floor.work_in_next_queue(job) for job in machine.queue
        ])

        s1 = operation_processing_time + next_remaining_processing_time
        s2 = (winq * 2 - 1) / operation_processing_time
        s3 = (winq + operation_processing_time + s1 / (winq - next_remaining_processing_time))
        s3 /= operation_processing_time

        values = s1 - s2 - s3

        return values
