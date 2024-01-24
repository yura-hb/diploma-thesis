import torch

from .scheduling_rule import SchedulingRule, MachineState, WaitInfo
from environment.job import Job


class RandomSchedulingRule(SchedulingRule):

    def __init__(self, generator: torch.Generator):
        super().__init__()

        self.generator = generator

    def __call__(self, machine_state: MachineState) -> Job | WaitInfo:
        index = torch.randint(0, len(machine_state.queue), (1,), generator=self.generator).item()

        return machine_state.queue[index]
