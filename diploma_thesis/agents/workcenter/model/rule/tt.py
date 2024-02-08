from .routing_rule import *


class TTRoutingRule(RoutingRule):
    """
    Shortest Total Waiting Time (TT) routing rule
    """

    def __call__(self, job: Job, work_center_idx: int, machines: List['Machine']) -> 'Machine | None':
        machine = min(machines, key=lambda machine: machine.cumulative_processing_time)

        return machine
