
from .scheduling_rule import *


class IdleSchedulingRule(SchedulingRule):

    def __call__(self, *args, **kwargs):
        return None

    @property
    def selector(self):
        return lambda x: None

    def criterion(self, machine: 'Machine', now: float) -> torch.FloatTensor:
        return torch.tensor(0.0)
