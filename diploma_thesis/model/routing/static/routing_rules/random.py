import torch

from environment import Job
from environment import Machine
from model.routing.static.routing_rules import RoutingRule

from typing import List


class RandomRoutingRule(RoutingRule):
    """
    Selects a machine at random
    """

    def __init__(self, generator: torch.Generator = torch.default_generator):
        super().__init__()

        self.generator = generator

    def select_machine(self, job: Job, work_center_idx: int, machines: List['Machine']) -> 'Machine | None':
        idx = torch.randint(0, len(machines), (1,), generator=self.generator)

        return machines[idx]
