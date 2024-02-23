from dataclasses import dataclass
from typing import List, Dict

import torch

from environment import Job, Machine
from .reward import MachineReward, RewardList


class GlobalDecomposedTardiness(MachineReward):
    """
    Reward from Deep-MARL external/PhD-Thesis-Projects/JSP/machine.py:648
    """

    @dataclass
    class Context:
        step_idx: int
        job: Job
        machine: Machine

    @dataclass
    class Configuration:
        exposure: float = 0.2
        sensitivity_to_slack: float = 200
        span: int = 128

        @staticmethod
        def from_cli(parameters: Dict) -> 'GlobalDecomposedTardiness.Configuration':
            return GlobalDecomposedTardiness.Configuration(
                exposure=parameters.get('exposure', 0.2),
                sensitivity_to_slack=parameters.get('sensitivity_to_slack', 200),
                span=parameters.get('span', 128)
            )

    def __init__(self, configuration: Configuration):
        super().__init__()

        self.configuration = configuration

    def record_job_action(self, job: Job, machine: Machine, moment: float) -> Context:
        return self.Context(job.current_step_idx, job, machine)

    def reward_after_production(self, context: Context) -> torch.FloatTensor | None:
        return None

    def reward_after_completion(self, contexts: List[Context]):
        work_center_idx = torch.tensor([c.step_idx for c in contexts])
        machine_idx = torch.tensor([c.job.history.arrived_machine_idx[c.step_idx] for c in contexts])
        reward = torch.zeros_like(work_center_idx, dtype=torch.float)
        job = contexts[0].job

        if job.is_tardy_upon_completion:
            wait_time = torch.FloatTensor([job.wait_time_on_machine(c.step_idx) for c in contexts])
            slack = torch.FloatTensor([job.slack_upon_arrival_on_machine(c.step_idx) for c in contexts])

            critical_factor = 1 - slack / (torch.abs(slack) + self.configuration.sensitivity_to_slack)
            exposure = self.configuration.exposure
            restructured_wait = wait_time * (1 - exposure) + torch.cat([wait_time[1:] * exposure, torch.zeros(1)])
            restructured_wait *= critical_factor

            reward = - torch.square(restructured_wait / self.configuration.span).clip(0, 1)

        return RewardList(work_center_idx=work_center_idx,
                          machine_idx=machine_idx,
                          reward=reward,
                          batch_size=work_center_idx.shape)

    @staticmethod
    def from_cli(parameters: Dict) -> MachineReward:
        return GlobalDecomposedTardiness(GlobalDecomposedTardiness.Configuration.from_cli(parameters))