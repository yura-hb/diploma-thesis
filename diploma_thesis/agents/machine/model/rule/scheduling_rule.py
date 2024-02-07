
import torch

from abc import ABCMeta, abstractmethod
from environment import Machine, Job, WaitInfo


class SchedulingRule(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, machine: 'Machine', now: float) -> Job | WaitInfo:
        pass
