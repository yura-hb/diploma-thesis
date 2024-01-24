from environment.job import Job
from routing_rules import WorkCenterState, RoutingRule


class SQRoutingRule(RoutingRule):
    """
    Selects machine with the shortest queue
    """

    def select_machine(self, job: Job, state: WorkCenterState) -> 'Machine':
        machines = state.machines
        machine = min(machines, key=lambda machine: len(machine.state.queue))

        return machine
