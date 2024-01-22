
from problem.job import Job
from routing_rule import RoutingRule
from routing_rules import WorkCenterState


class EARoutingRule(RoutingRule):

    def select_machine(self, job: Job, state: WorkCenterState) -> int:
        ...