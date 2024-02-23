from dataclasses import dataclass
from typing import List, Dict

import torch

from environment import Job, Machine
from .reward import MachineReward, RewardList


class GlobalTardiness(MachineReward):
    """
    Reward from Deep-MARL external/PhD-Thesis-Projects/JSP/machine.py:693
    """

    @dataclass
    class Context:
        step_idx: int
        job: Job
        machine: Machine

    @dataclass
    class Configuration:

        span: int = 256

        @staticmethod
        def from_cli(parameters: Dict) -> 'GlobalTardiness.Configuration':
            return GlobalTardiness.Configuration(
                span=parameters.get('span', 256)
            )

    def __init__(self, configuration: Configuration):
        self.configuration = configuration

    def record_job_action(self, job: Job, machine: Machine, moment: float) -> Context:
        return self.Context(job.current_step_idx, job, machine)

    def reward_after_production(self, context: Context) -> torch.FloatTensor | None:
        return None

    def reward_after_completion(self, contexts: List[Context]):
        work_center_idx = torch.tensor([c.step_idx for c in contexts])
        machine_idx = torch.tensor([c.job.history.arrived_machine_idx[c.step_idx] for c in contexts])
        reward = torch.zeros_like(work_center_idx, dtype=torch.float)

        if contexts[0].job.is_tardy_upon_completion:
            tardy_rate = - torch.clip(contexts[0].job.tardiness_upon_completion / self.configuration.span, 0, 1)

            reward += tardy_rate

        return RewardList(work_center_idx=work_center_idx,
                          machine_idx=machine_idx,
                          reward=reward,
                          batch_size=work_center_idx.shape)

    @staticmethod
    def from_cli(parameters) -> MachineReward:
        return GlobalTardiness(GlobalTardiness.Configuration.from_cli(parameters))
