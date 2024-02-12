
import torch

from abc import ABCMeta, abstractmethod
from environment import Machine, Job, WaitInfo, JobReductionStrategy


class SchedulingRule(metaclass=ABCMeta):

    def __init__(self, reduction_strategy: JobReductionStrategy = JobReductionStrategy.mean):
        self.reduction_strategy = reduction_strategy

    @abstractmethod
    def __call__(self, machine: 'Machine', now: float) -> Job | WaitInfo:
        pass
