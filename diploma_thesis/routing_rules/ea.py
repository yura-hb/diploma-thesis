
from environment.job import Job
from routing_rules import WorkCenterState, RoutingRule


class EARoutingRule(RoutingRule):
    """
    Earliest Available (EA) routing rule
    """

    def select_machine(self, job: Job, state: WorkCenterState) -> 'Machine':
        machines = state.machines
        machine = min(machines, key=lambda machine: machine.time_till_available)

        return machine
