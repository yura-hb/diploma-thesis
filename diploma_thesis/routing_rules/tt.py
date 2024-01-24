from environment.job import Job
from routing_rules import WorkCenterState, RoutingRule


class TTRoutingRule(RoutingRule):
    """
    Shortest Total Waiting Time (TT) routing rule
    """

    def select_machine(self, job: Job, state: WorkCenterState) -> 'Machine':
        machines = state.machines
        machine = min(machines, key=lambda machine: machine.cumulative_processing_time)

        return machine
