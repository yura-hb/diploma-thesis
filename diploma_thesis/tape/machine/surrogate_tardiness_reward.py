from dataclasses import dataclass
from typing import List

import torch

from environment import Job, Machine, JobReductionStrategy
from .reward import MachineReward, RewardList


class SurrogateTardinessReward(MachineReward):
    """
    Reward from Deep-MARL external/PhD-Thesis-Projects/JSP/machine.py:693
    """

    @dataclass
    class Context:
        work_center_idx: torch.LongTensor
        machine_idx: torch.LongTensor
        processing_time: torch.FloatTensor
        slack: torch.FloatTensor
        winq: torch.FloatTensor
        chosen: torch.LongTensor

    @dataclass
    class Configuration:
        critical_level_factor: float = 64
        winq_factor: float = 0.2
        span: int = 20
        release_reward_after_completion: bool = False

        @staticmethod
        def from_cli(parameters: dict) -> 'SurrogateTardinessReward.Configuration':
            return SurrogateTardinessReward.Configuration(
                critical_level_factor=parameters.get('critical_level_factor', 64),
                winq_factor=parameters.get('winq_factor', 0.2),
                span=parameters.get('span', 20),
                release_reward_after_completion=parameters.get('release_reward_after_completion', False)
            )

    def __init__(self,
                 strategy: JobReductionStrategy = JobReductionStrategy.mean,
                 configuration: Configuration = Configuration()):
        super().__init__()

        self.strategy = strategy
        self.configuration = configuration

    def record_job_action(self, job: Job, machine: Machine, moment: float) -> Context:
        return self.Context(
            work_center_idx=machine.work_center_idx,
            machine_idx=machine.machine_idx,
            processing_time=torch.FloatTensor([
                job.current_operation_processing_time_on_machine for job in machine.queue
            ]),
            slack=torch.FloatTensor([
                job.slack_upon_moment(0, self.strategy) for job in machine.queue
            ]),
            winq=torch.FloatTensor([
                machine.shop_floor.work_in_next_queue(job) for job in machine.queue
            ]),
            chosen=torch.tensor([job.id == _job.id for _job in machine.queue])
        )

    def reward_after_production(self, context: Context) -> torch.FloatTensor | None:
        if not self.configuration.release_reward_after_completion:
            return self.__compute_reward__(context)

    def reward_after_completion(self, contexts: List[Context]):
        if not self.configuration.release_reward_after_completion:
            return None

        return RewardList(
            work_center_idx=torch.stack([context.work_center_idx for context in contexts]),
            machine_idx=torch.stack([context.machine_idx for context in contexts]),
            reward=torch.stack([self.__compute_reward__(context) for context in contexts]),
        )

    def __compute_reward__(self, context: Context):
        slack = context.slack

        critical_level = 1 - slack / (torch.abs(slack) + self.configuration.critical_level_factor)

        critical_level_chosen = critical_level[context.chosen]
        critical_level_loser = critical_level[~context.chosen]

        earned_slack_chosen = torch.mean(context.processing_time[~context.chosen])
        earned_slack_chosen *= critical_level_chosen

        consumed_slack_loser = context.processing_time[context.chosen] * critical_level_loser.mean()

        reward_slack = earned_slack_chosen - consumed_slack_loser
        reward_winq = (
            context.winq[~context.chosen].mean() - context.winq[context.chosen]
        ) * self.configuration.winq_factor

        reward = ((reward_slack + reward_winq) / self.configuration.span).clip(-1, 1)

        return reward

    @staticmethod
    def from_cli(parameters) -> MachineReward:
        return SurrogateTardinessReward(SurrogateTardinessReward.Configuration.from_cli(parameters))
