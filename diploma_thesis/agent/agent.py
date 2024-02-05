from typing import List
from environment import Job, WorkCenter, Machine, WaitInfo
from dataclasses import dataclass

import environment

from typing import Any


class Agent(environment.Agent):

    @dataclass
    class Configuration:
        # Variants
        #   - Episodic (Train After Each Episode)
        #   - Parameters:
        #      - Warm up steps
        #      - Duration of episode
        #      -
        #   - Continuous
        #   - Parameters:
        #      - Warm up steps
        #      - Duration of simulation
        #      - Intervals between training steps
        simulation: 'str'

    @dataclass
    class Module:
        state_encoder: Any
        brain: Any
        reward: Any

    def __init__(self, machine_module: Module, work_center_module: Module):
        super().__init__()

        self.machine_module = machine_module
        self.work_center_module = work_center_module

    def learn(self):
        pass

    def evaluate(self) -> environment.ShopFloor:
        pass

    def save(self):
        pass

    @staticmethod
    def load(path: str) -> 'Agent':
        pass

    # Shop Floor Events

    def schedule(self, shop_floor_id: int, machine: Machine, now: int) -> Job | WaitInfo:
        pass

    def route(self, shop_floor_id: int, job: Job, work_center_idx: int, machines: List['Machine']) -> 'Machine | None':
        pass

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
