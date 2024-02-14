
from abc import ABCMeta, abstractmethod
from typing import TypeVar, List

import torch

from environment import Job, Machine, ShopFloor
from dataclasses import dataclass

Context = TypeVar('Context')


class MachineReward(metaclass=ABCMeta):

    @abstractmethod
    def record_job_action(self, job: Job, machine: Machine) -> Context:
        """
        An action to be called when reward model need to capture metrics at the moment of job selection
        """
        pass

    @abstractmethod
    def reward_after_production(self, context: Context) -> torch.FloatTensor | None:
        """

        """
        pass

    @abstractmethod
    def reward_after_completion(self, context: List[Context]) -> torch.FloatTensor | None:
        """
        Returns: A tensor for each machine in job path, i.e. of dim (Job.step_idx.shape) or none if reward
                 can't be computed
        """
        pass
