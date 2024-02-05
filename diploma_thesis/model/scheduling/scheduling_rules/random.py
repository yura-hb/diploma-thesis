import torch

import environment
from model.scheduling.scheduling_rule import SchedulingRule, WaitInfo


class RandomSchedulingRule(SchedulingRule):

    def __init__(self, generator: torch.Generator = torch.default_generator):
        super().__init__()

        self.generator = generator

    def __call__(self, machine: environment.Machine, now: float) -> environment.Job | WaitInfo:
        index = torch.randint(0, len(machine.queue), (1,), generator=self.generator).item()

        return machine.queue[index]
