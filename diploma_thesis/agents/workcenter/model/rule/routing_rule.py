from typing import List

import torch

from abc import ABCMeta, abstractmethod
from environment import Job, Machine, JobReductionStrategy


class RoutingRule(metaclass=ABCMeta):
    """
    Selects a machine at random
    """

    @abstractmethod
    def __call__(self, job: Job, work_center_idx: int, machines: List[Machine]) -> Machine | None:
        pass

