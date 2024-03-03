
from abc import ABCMeta, abstractmethod
from typing import TypeVar, List

import torch
from tensordict.prototype import tensorclass

from environment import Job, Machine


@tensorclass
class RewardList:
    # Indices of contexts provided to reward_after_completion
    indices: torch.LongTensor
    # Reward for each context
    reward: torch.FloatTensor
    # Ids of affected units as tensor (2, n), where first row is work_center_idx and second machine_idx
    units: torch.LongTensor


class MachineReward(metaclass=ABCMeta):

    Context = TypeVar('Context')

    @abstractmethod
    def record_job_action(self, job: Job | None, machine: Machine, moment: float) -> Context:
        """
        An action to be called when reward policy need to capture metrics at the moment of job selection
        """
        pass

    @abstractmethod
    def reward_after_production(self, context: Context) -> torch.FloatTensor | None:
        """
        Returns: A reward for step represented by context
        """
        pass

    @abstractmethod
    def reward_after_completion(self, contexts: List[Context]) -> RewardList | None:
        """
        Returns: A tensor for each machine in job path, i.e. of dim (Job.step_idx.shape) or none if reward
                 can't be computed
        """
        pass
