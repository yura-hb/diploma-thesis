from typing import List

from environment import Job, Machine
from model.routing.static.routing_rules import RoutingRule


class TTRoutingRule(RoutingRule):
    """
    Shortest Total Waiting Time (TT) routing rule
    """

    def select_machine(self, job: Job, work_center_idx: int, machines: List['Machine']) -> 'Machine | None':
        machine = min(machines, key=lambda machine: machine.cumulative_processing_time)

        return machine
