from problem.job import Job
from routing_rule import RoutingRule
from routing_rules import WorkCenterState


class TTRoutingRule(RoutingRule):
    """
    Shortest Total Waiting Time (TT) routing rule
    """

    def __init__(self):
        super().__init__()

    def select_machine(self, job: Job, state: WorkCenterState) -> int:
        ...
