from .routing_rule import *


class IdleRoutingRule(RoutingRule):
    """
    Selects a machine at random
    """

    def __call__(self, *args, **kwargs):
        return None

    @property
    def selector(self):
        return lambda x: None

    def criterion(self, job: Job, work_center: WorkCenter) -> torch.FloatTensor:
        return torch.tensor(0.0)
