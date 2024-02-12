from .scheduling_rule import *


class WINQSchedulingRule(SchedulingRule):
    """
    Work In Next Queue scheduling rule
    """

    def __call__(self, machine: Machine, now: float) -> Job | WaitInfo:
        values = torch.FloatTensor([machine.shop_floor.work_in_next_queue(job) for job in machine.queue])
        idx = torch.argmin(values)

        return machine.queue[idx]
