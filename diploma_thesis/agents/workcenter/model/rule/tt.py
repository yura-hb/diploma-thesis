from .routing_rule import *


class TTRoutingRule(RoutingRule):
    """
    Shortest Total Waiting Time (TT) routing rule
    """

    @property
    def selector(self):
        return torch.argmin

    def criterion(self, job: Job, work_center: WorkCenter) -> torch.FloatTensor:
        return torch.FloatTensor([machine.cumulative_processing_time for machine in work_center.machines])
