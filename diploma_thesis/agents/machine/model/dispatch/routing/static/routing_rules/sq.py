from typing import List

from environment import Job, Machine, RoutingRule


class SQRoutingRule(RoutingRule):
    """
    Selects machine with the shortest queue
    """

    def select_machine(self, job: Job, work_center_idx: int, machines: List['Machine']) -> 'Machine | None':
        machine = min(machines, key=lambda machine: len(machine.state.queue))

        return machine
