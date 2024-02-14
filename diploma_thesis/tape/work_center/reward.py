
from abc import ABCMeta, abstractmethod
from typing import TypeVar

import torch

from typing import List
from environment import Job, WorkCenter, ShopFloor
from dataclasses import dataclass

Context = TypeVar('Context')


class WorkCenterReward(metaclass=ABCMeta):

    @abstractmethod
    def record_job_action(self, job: Job, work_center: WorkCenter) -> Context:
        pass

    @abstractmethod
    def reward_after_production(self, context: Context) -> torch.FloatTensor | None:
        pass

    @abstractmethod
    def reward_after_completion(self, context: List[Context]) -> torch.FloatTensor | None:
        pass
