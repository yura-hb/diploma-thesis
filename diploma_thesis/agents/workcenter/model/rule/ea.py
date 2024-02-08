from .routing_rule import *


class EARoutingRule(RoutingRule):
    """
    Earliest Available (EA) routing rule
    """

    def __call__(self, job: Job, work_center_idx: int, machines: List['Machine']) -> 'Machine | None':
        machine = min(machines, key=lambda machine: machine.time_till_available)

        return machine
