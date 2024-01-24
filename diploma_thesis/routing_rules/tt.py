from environment.job import Job
from routing_rules import WorkCenterState, RoutingRule


class TTRoutingRule(RoutingRule):
    """
    Shortest Total Waiting Time (TT) routing rule
    """

    def select_machine(self, job: Job, state: WorkCenterState) -> int:
        ...
