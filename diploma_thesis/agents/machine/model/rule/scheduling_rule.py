
import torch

from abc import ABCMeta, abstractmethod
from environment import Machine, Job, WaitInfo, JobReductionStrategy


class SchedulingRule(metaclass=ABCMeta):

    def __init__(self, reduction_strategy: JobReductionStrategy = JobReductionStrategy.mean):
        self.reduction_strategy = reduction_strategy

    def __call__(self, machine: 'Machine', now: float) -> Job | WaitInfo:
        value = self.criterion(machine, now)
        idx = self.selector(value)

        return machine.queue[idx]

    @property
    @abstractmethod
    def selector(self):
        pass

    @abstractmethod
    def criterion(self, machine: 'Machine', now: float) -> torch.FloatTensor:
        pass
