from typing import List

import torch

from environment import Job, Machine, RoutingRule


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
