from .routing_rule import *


class UTRoutingRule(RoutingRule):
    """
    Selects machine with the lowest utilization
    """

    def __call__(self, job: Job, work_center_idx: int, machines: List['Machine']) -> 'Machine | None':
        machine = min(machines, key=lambda machine: machine.cumulative_run_time)

        return machine
