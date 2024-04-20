from dataclasses import dataclass
from typing import List, Dict

import torch

from environment import Job, Machine
from .reward import MachineReward, RewardList


class Unary(MachineReward):
    """
    Reward from Deep-MARL external/PhD-Thesis-Projects/JSP/machine.py:693
    """

    @dataclass
    class Context:
        step_idx: int
        machine: Machine
        expected_tardiness_rate: float
        actual_tardiness_rate: float
        utilization_rate: float

    @dataclass
    class Configuration:

        @staticmethod
        def from_cli(parameters: Dict) -> 'Unary.Configuration':
            return Unary.Configuration()

    def __init__(self, configuration: Configuration):
        self.configuration = configuration

    def record_job_action(self, job: Job, machine: Machine, moment: float) -> Context:
        return self.Context(
            job.current_step_idx,
            machine=machine,
            expected_tardiness_rate=machine.shop_floor.expected_tardy_rate(moment),
            actual_tardiness_rate=machine.shop_floor.actual_tardy_rate(moment),
            utilization_rate=machine.shop_floor.utilization_rate()
        )

    def reward_after_production(self, context: Context) -> torch.FloatTensor | None:
        now = context.machine.shop_floor.now
        actual_tardiness_rate = context.machine.shop_floor.actual_tardy_rate(now)

        if actual_tardiness_rate < context.actual_tardiness_rate:
            return torch.FloatTensor(1)

        if actual_tardiness_rate > context.actual_tardiness_rate:
            return torch.FloatTensor(-1)

        expected_tardiness_rate = context.machine.shop_floor.expected_tardy_rate(now)

        if expected_tardiness_rate < context.expected_tardiness_rate:
            return torch.FloatTensor(1)

        if expected_tardiness_rate > context.expected_tardiness_rate:
            return torch.FloatTensor(-1)

        utilization_rate = context.machine.shop_floor.utilization_rate()

        if utilization_rate < context.utilization_rate:
            return torch.FloatTensor(1)

        if utilization_rate > context.utilization_rate * 0.95:
            return torch.FloatTensor(0)

        return torch.FloatTensor(-1)

    def reward_after_completion(self, contexts: List[Context]):
        return None

    @staticmethod
    def from_cli(parameters) -> MachineReward:
        return Unary(Unary.Configuration.from_cli(parameters))
