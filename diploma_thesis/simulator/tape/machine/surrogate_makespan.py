

from dataclasses import dataclass
from typing import List, Dict

import torch

from environment import Job, Machine, JobReductionStrategy
from .reward import MachineReward, RewardList

# Reward from
# Hierarchical Reinforcement Learning for Dynamic Job Shop Scheduling


class SurrogateMakespan(MachineReward):

    @dataclass
    class Context:
        work_center_idx: torch.LongTensor
        machine_idx: torch.LongTensor
        completion_lower_bound: torch.FloatTensor

    @dataclass
    class Configuration:
        release_reward_after_completion: bool = False

        @staticmethod
        def from_cli(parameters: Dict) -> 'SurrogateMakespan.Configuration':
            return SurrogateMakespan.Configuration(
                release_reward_after_completion=parameters.get('release_reward_after_completion', False)
            )

    def __init__(self, configuration: Configuration, strategy: JobReductionStrategy = JobReductionStrategy.mean):
        super().__init__()

        self.configuration = configuration
        self.strategy = strategy

    def record_job_action(self, job: Job, machine: Machine, moment: float) -> Context:
        # TODO: Implement
        return self.Context(
            work_center_idx=machine.work_center_idx,
            machine_idx=machine.machine_idx,
            completion_time=...
        )

    def reward_after_production(self, context: Context) -> torch.FloatTensor | None:
        if not self.configuration.release_reward_after_completion:
            return self.__compute_reward__(context)

    def reward_after_completion(self, contexts: List[Context]):
        if not self.configuration.release_reward_after_completion:
            return None

        work_center_idx = torch.tensor([c.work_center_idx for c in contexts])
        machine_idx = torch.tensor([c.machine_idx for c in contexts])

        return RewardList(
            indices=torch.arange(len(contexts)),
            units=torch.vstack([work_center_idx, machine_idx]),
            reward=torch.stack([self.__compute_reward__(context) for context in contexts]),
            batch_size=[]
        )

    def __compute_reward__(self, context: Context):
        return - context.slack.mean() / (context.remaining_processing_time.mean() + 0.01)

    @staticmethod
    def from_cli(parameters: Dict) -> MachineReward:
        return SurrogateSlack(SurrogateSlack.Configuration.from_cli(parameters))
