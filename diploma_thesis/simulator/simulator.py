
import simpy

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from environment import Delegate, Agent, Machine, Job, WaitInfo
from typing import List
from agents import WorkCenterAgent, MachineAgent, TrainingPhase, EvaluationPhase, WarmUpPhase, Phase
from .reward import RewardModel


class Simulator(Agent, Delegate, metaclass=ABCMeta):

    @dataclass
    class Configuration:
        warm_up_phases: List[float]
        pretrain_steps: int


    def __init__(self,
                 work_center: WorkCenterAgent,
                 machine: MachineAgent,
                 reward_model: RewardModel):
        self.work_center = work_center
        self.machine = machine
        self.reward_model = reward_model
        self.environment = simpy.Environment()

    def simulate(self):
        for index, phase in enumerate(self.configuration.warm_up_phases):
            self.__update__(WarmUpPhase(index))
            self.environment.run(until=phase)

        self.__update__(TrainingPhase())

        yield self.environment.process(self.run())

        self.__update__(EvaluationPhase())

    @abstractmethod
    def run(self):
        pass

    def schedule(self, shop_floor_id: int, machine: Machine, now: int) -> Job | WaitInfo:
        state = self.machine.encode_state(machine)

        return self.machine.schedule(state)

    def route(self, shop_floor_id: int, job: Job, work_center_idx: int, machines: List[Machine]) -> 'Machine | None':
        state = self.work_center.encode_state(work_center_idx, machines)

        return self.work_center.schedule(state)

    def __update__(self, phase: Phase):
        self.work_center.update(phase)
        self.machine.update(phase)
