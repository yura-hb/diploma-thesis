from .scheduling_rule import *


class DPTWINQNPTSchedulingRule(SchedulingRule):
    """
    Double Processing Time + Work In Next Queue + Next Processing Time
    """

    def __call__(self, machine: Machine, now: float) -> Job | WaitInfo:
        processing_times = torch.FloatTensor(
            [job.current_operation_processing_time_on_machine for job in machine.queue]
        )
        winq = torch.FloatTensor(
            [machine.shop_floor.work_in_next_queue(job) for job in machine.queue]
        )
        next_operation_processing_times = torch.FloatTensor(
            [job.next_operation_processing_time(self.reduction_strategy) for job in machine.queue]
        )

        result = 2 * processing_times + winq + next_operation_processing_times
        index = torch.argmin(result)

        return machine.queue[index]
