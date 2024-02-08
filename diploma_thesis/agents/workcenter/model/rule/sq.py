from .routing_rule import *


class SQRoutingRule(RoutingRule):
    """
    Selects machine with the shortest queue
    """

    def __call__(self, job: Job, work_center_idx: int, machines: List['Machine']) -> 'Machine | None':
        machine = min(machines, key=lambda machine: len(machine.state.queue))

        return machine
