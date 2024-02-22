from .routing_rule import *


class UTRoutingRule(RoutingRule):
    """
    Selects machine with the lowest utilization
    """

    @property
    def selector(self):
        return torch.argmin

    def criterion(self, job: Job, work_center: WorkCenter) -> torch.FloatTensor:
        return torch.FloatTensor([machine.cumulative_run_time for machine in work_center.machines])
