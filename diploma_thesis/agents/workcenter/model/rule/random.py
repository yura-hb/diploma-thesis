from .routing_rule import *


class RandomRoutingRule(RoutingRule):
    """
    Selects a machine at random
    """

    @property
    def selector(self):
        return lambda x: torch.randint(0, len(x), (1,))

    def criterion(self, job: Job, work_center: WorkCenter) -> torch.FloatTensor:
        return torch.zeros(len(work_center.machines))

