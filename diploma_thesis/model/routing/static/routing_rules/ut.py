from environment import Machine, Job
from typing import List
from .routing_rule import RoutingRule


class UTRoutingRule(RoutingRule):
    """
    Selects machine with the lowest utilization
    """

    def select_machine(self, job: Job, work_center_idx: int, machines: List['Machine']) -> 'Machine | None':
        machine = min(machines, key=lambda machine: machine.cumulative_run_time)

        return machine
