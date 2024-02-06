import simpy

from .simulator import Simulator
from environment import Statistics, Job, WorkCenter, Machine, WaitInfo
from agents import WorkCenterAgent, MachineAgent
from dataclasses import dataclass
from typing import Any, List


class EpisodicSimulator(Simulator):
    """
    A simulator, which launches several shop=floors in parallel and simulates them until terminating conditions are met.
    During the process of the simulation the whole episode of environment is recorded.

    After the simulation is finished returns are estimated and passed to the agent for training.
    """

    @dataclass
    class Configuration:
        episode_count: int
        parallel_environments: int
        terminating_condition: 'Any'
        return_estimation: 'Any'

    def __init__(self,
                 work_center: WorkCenterAgent,
                 machine: MachineAgent,
                 reward_model,
                 configuration: Configuration):
        super().__init__(work_center, machine, reward_model)

        self.configuration = configuration

    def simulate(self) -> [Statistics]:
        environment = simpy.Environment()

        self.environment.process()

    def will_produce(self, shop_floor_id: int, job: Job, machine: Machine):
        pass

    def did_produce(self, shop_floor_id: int, job: Job, machine: Machine):
        pass

    def will_dispatch(self, shop_floor_id: int, job: Job, work_center: WorkCenter):
        pass

    def did_dispatch(self, shop_floor_id: int, job: Job, work_center: WorkCenter, machine: Machine):
        pass

    def did_finish_dispatch(self, shop_floor_id: int, work_center: WorkCenter):
        pass

    def did_complete(self, shop_floor_id: int, job: Job):
        pass
