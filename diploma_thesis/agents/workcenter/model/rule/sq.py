from .routing_rule import *


class SQRoutingRule(RoutingRule):
    """
    Selects machine with the shortest queue
    """

    @property
    def selector(self):
        return torch.argmin

    def criterion(self, job: Job, work_center: WorkCenter) -> torch.FloatTensor:
        return torch.FloatTensor([len(machine.queue) for machine in work_center.machines])
