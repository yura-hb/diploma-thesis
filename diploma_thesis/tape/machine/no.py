from dataclasses import dataclass
from typing import List

import torch

from environment import Job, Machine
from .reward import MachineReward, RewardList


class No(MachineReward):

    @dataclass
    class Context:
        pass

    def record_job_action(self, job: Job, machine: Machine, moment: float) -> Context:
        return self.Context()

    def reward_after_production(self, context: Context) -> torch.FloatTensor | None:
        """
        Returns: A tensor for
        """
        return None

    def reward_after_completion(self, contexts: List[Context]) -> RewardList | None:
        return None

    @staticmethod
    def from_cli(parameters) -> MachineReward:
        return No()
