from .routing_rule import *


class EARoutingRule(RoutingRule):
    """
    Earliest Available (EA) routing rule
    """

    @property
    def selector(self):
        return torch.argmin

    def criterion(self, job: Job, work_center: WorkCenter) -> torch.FloatTensor:
        return torch.FloatTensor([machine.time_till_available for machine in work_center.machines])
