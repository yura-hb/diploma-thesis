import torch

from environment.job import Job
from routing_rules import WorkCenterState, RoutingRule
from environment.machine import Machine


class RandomRoutingRule(RoutingRule):
    """
    Selects a machine at random
    """

    def __init__(self, generator: torch.Generator):
        super().__init__()

        self.generator = generator

    def select_machine(self, job: Job, state: WorkCenterState) -> 'Machine':
        machines = state.machines

        idx = torch.randint(0, len(machines), (1,), generator=self.generator)

        return machines[idx]
