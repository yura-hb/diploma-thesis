from .scheduling_rule import *


class PTWINQSSchedulingRule(SchedulingRule):
    """
    Processing Time + Work In Next Queue + Slack
    """

    def __call__(self, machine: Machine, now: float) -> Job | WaitInfo:
        processing_time = torch.FloatTensor([
            job.current_operation_processing_time_on_machine for job in machine.queue
        ])
        winq = torch.FloatTensor([
            machine.shop_floor.work_in_next_queue(job) for job in machine.queue
        ])
        slack = torch.FloatTensor([
            job.slack_upon_moment(now, self.reduction_strategy) for job in machine.queue
        ])
        idx = torch.argmin(winq + processing_time + slack)

        return machine.queue[idx]
