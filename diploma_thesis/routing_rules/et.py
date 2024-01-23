from environment.job import Job
from routing_rules import WorkCenterState, RoutingRule


class ETRoutingRule(RoutingRule):

    def select_machine(self, job: Job, state: WorkCenterState) -> int:
        ...