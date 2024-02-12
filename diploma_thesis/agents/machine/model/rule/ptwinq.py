from .scheduling_rule import *


class PTWINQSchedulingRule(SchedulingRule):
    """
    Processing Time + Work In Next Queue
    """

    def __call__(self, machine: Machine, now: float) -> Job | WaitInfo:
        processing_time = torch.FloatTensor([
            job.current_operation_processing_time_on_machine for job in machine.queue
        ])
        winq = torch.FloatTensor([
            machine.shop_floor.work_in_next_queue(job) for job in machine.queue
        ])
        idx = torch.argmin(winq + processing_time)

        return machine.queue[idx]
