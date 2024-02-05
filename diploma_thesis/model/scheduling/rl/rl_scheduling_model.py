from abc import ABCMeta, abstractmethod

import simpy
import torch

import environment
from model.scheduling import scheduling_rules

from dataclasses import dataclass
from model.scheduling.scheduling_model import SchedulingModel
from typing import List


class InnerModel(metaclass=ABCMeta):

    @dataclass
    class Record:
        initial_state: torch.FloatTensor
        action: torch.LongTensor
        next_state: torch.FloatTensor
        reward: torch.FloatTensor

    @abstractmethod
    def make_state(self, machine_state: environment.Machine) -> torch.FloatTensor:
        pass

    @abstractmethod
    def training_step(self):
        pass


class RLSchedulingModel(SchedulingModel, environment.ShopFloor.Delegate):
    """
    Base Class representing reinforcement learning pipeline
    """

    class Phase:
        pass

    @dataclass
    class WarmUpPhase(Phase):
        step: int

    @dataclass
    class TrainingPhase(Phase):
        pass

    @dataclass
    class EvaluationPhase(Phase):
        pass

    @dataclass
    class Configuration:
        # List of durations of each warm-up phase
        warm_up_phases: List[float]
        # Number of steps to pretrain the model after warm-up
        pretrain_steps: int
        # Duration of interval to trigger training of the model
        train_interval: int
        # Number of steps to update the model
        training_steps: int
        # Problem
        problem: environment.Problem

    def __init__(self, configuration: Configuration, inner_model: InnerModel, environment: simpy.Environment):
        self.configuration = configuration
        self.environment = environment
        self.inner_model = inner_model
        self.phase = self.EvaluationPhase()

        super().__init__()

        self.environment.process(self.__warm_up__())
        self.environment.process(self.__train__())

    def update(self, phase: Phase):
        self.phase = phase

    def __warm_up__(self):
        for step, duration in enumerate(self.configuration.warm_up_phases):
            yield self.environment.timeout(duration)
            self.update(self.WarmUpPhase(step))

    def __train__(self):
        total_warm_up_duration = sum(self.configuration.warm_up_phases)

        yield self.environment.timeout(total_warm_up_duration)

        self.update(self.TrainingPhase())

        for i in range(self.configuration.pretrain_steps):
            self.inner_model.training_step()

        while self.environment.now < self.configuration.problem.timespan:
            yield self.environment.timeout(self.configuration.train_interval)
            self.inner_model.training_step()

        self.update(self.EvaluationPhase())

    def __call__(self, machine: environment.Machine, now: float) -> environment.Job | scheduling_rules.WaitInfo:
        pass

    def will_produce(self, job: environment.Job, machine: environment.Machine):
        if isinstance(self.phase, self.EvaluationPhase):
            return

    def did_produce(self, job: environment.Job, machine: environment.Machine):
        if isinstance(self.phase, self.EvaluationPhase):
            return

    def did_complete(self, job: environment.Job):
        if isinstance(self.phase, self.EvaluationPhase):
            return

    def will_dispatch(self, job: environment.Job, work_center: environment.WorkCenter):
        pass
