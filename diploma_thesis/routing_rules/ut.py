from environment.job import Job
from routing_rules import WorkCenterState, RoutingRule


class UTRoutingRule(RoutingRule):
    """
    Selects machine with the lowest utilization
    """

    def select_machine(self, job: Job, state: WorkCenterState) -> 'Machine':
        machines = state.machines
        machine = min(machines, key=lambda machine: machine.cumulative_run_time)

        return machine
