from dataclasses import dataclass
from typing import List, Dict

import torch

from environment import Job, Machine
from .reward import MachineReward


class Makespan(MachineReward):
    """
    Reward from Deep-MARL external/PhD-Thesis-Projects/JSP/machine.py:693
    """

    @dataclass
    class Context:
        lower_bound: int
        jobs: List[Job]
        machine: Machine

    @dataclass
    class Configuration:
        span: int = 256
        clip: bool = False

        # @staticmethod
        def from_cli(parameters: Dict) -> 'Makespan.Configuration':
            return Makespan.Configuration(span=parameters.get('span', 256), clip=parameters.get('clip', False))

    def __init__(self, configuration: Configuration):
        self.configuration = configuration

    def record_job_action(self, job: Job, machine: Machine, moment: float) -> Context:
        jobs = machine.shop_floor.in_system_running_jobs
        lower_bound = max([job.estimated_completion_time(machine.shop_floor.now) for job in jobs])

        return self.Context(lower_bound, jobs, machine)

    def reward_after_production(self, context: Context) -> torch.FloatTensor | None:
        lower_bound = max([job.estimated_completion_time(context.machine.shop_floor.now) for job in context.jobs])

        delta = context.lower_bound - lower_bound
        delta /= self.configuration.span

        if self.configuration.clip:
            delta = delta.clip(-1, 1)

        return delta.squeeze()

    def reward_after_completion(self, contexts: List[Context]):
        return None

    @staticmethod
    def from_cli(parameters) -> MachineReward:
        return Makespan(Makespan.Configuration.from_cli(parameters))


