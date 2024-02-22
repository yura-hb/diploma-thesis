
import torch

from abc import ABCMeta, abstractmethod
from environment import Job, Machine, WorkCenter, JobReductionStrategy


class RoutingRule(metaclass=ABCMeta):
    """
    Selects a machine at random
    """
    def __call__(self, job: Job, work_center: WorkCenter) -> Machine | None:
        value = self.criterion(job, work_center)
        selector = self.selector
        idx = selector(value)

        return work_center.machines[idx]

    @property
    @abstractmethod
    def selector(self):
        pass

    @abstractmethod
    def criterion(self, job: Job, work_center: WorkCenter) -> torch.FloatTensor:
        pass
