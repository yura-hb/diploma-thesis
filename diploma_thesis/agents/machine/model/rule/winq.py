from .scheduling_rule import *


class WINQSchedulingRule(SchedulingRule):
    """
    Work In Next Queue scheduling rule
    """

    def selector(self):
        return torch.argmin

    def criterion(self, machine: Machine, now: float) -> torch.FloatTensor:
        values = torch.FloatTensor([machine.shop_floor.work_in_next_queue(job) for job in machine.queue])

        return values
