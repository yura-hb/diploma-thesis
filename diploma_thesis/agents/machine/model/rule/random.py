from .scheduling_rule import *


class RandomSchedulingRule(SchedulingRule):

    def __call__(self, machine: Machine, now: float) -> Job | WaitInfo:
        index = torch.randint(0, len(machine.queue), (1,)).item()

        return machine.queue[index]
