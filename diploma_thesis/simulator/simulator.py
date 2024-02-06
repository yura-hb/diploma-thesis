import logging
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import List, Callable

import simpy

from agents import WorkCenterAgent, MachineAgent, TrainingPhase, EvaluationPhase, WarmUpPhase, Phase
from environment import Delegate, Agent
from .reward import RewardModel
from .simulation import Simulation


@dataclass
class TimelineSchedule:
    warm_up_phases: List[float] = field(default_factory=list)
    max_length: int = 1000


@dataclass
class TrainSchedule:
    pretrain_steps: int = 0
    train_interval: int = 0
    max_training_steps: int = 0


@dataclass
class RunSchedule:
    n_workers: int = 4
    simulations: List[Simulation] = field(default_factory=list)


@dataclass
class EvaluateSchedule:
    n_workers: int = 4
    simulations: List[Simulation] = field(default_factory=list)


class Simulator(Agent, Delegate, metaclass=ABCMeta):

    def __init__(
        self, work_center: WorkCenterAgent, machine: MachineAgent, reward_model: RewardModel, logger: logging.Logger
    ):
        self.work_center = work_center
        self.machine = machine
        self.reward_model = reward_model
        self.logger = logger
        self.environment = simpy.Environment()

    def train(
        self, timeline: TimelineSchedule, machine_train_schedule: TrainSchedule,
        work_center_train_schedule: TrainSchedule, run_schedule: RunSchedule
    ):
        assert self.machine.is_trainable or self.work_center.is_trainable, 'At least one agent should be trainable'

        env = self.environment
        warmup_end = env.event()
        machine_training_end = env.event()
        work_center_train_end = env.event()
        run_end = env.event()
        all_of_event = simpy.AllOf(env, [warmup_end, machine_training_end, work_center_train_end, run_end])

        env.process(self.__main_timeline__(timeline, warmup_end, run_end))

        ids = ['machine', 'work_center']
        schedules = [machine_train_schedule, work_center_train_schedule]
        end_training_events = [machine_training_end, work_center_train_end]
        train_steps = [self.machine.train_step, self.work_center.train_step]

        for idx, schedule, end_of_train_event, train_step in zip(ids, schedules, end_training_events, train_steps):
            env.process(self.__train_timeline__(idx, schedule, warmup_end, end_of_train_event, run_end, train_step))

        env.process(self.__run__(run_event=run_end, configuration=run_schedule))
        env.process(self.__terminate_if_needed__(run_event=run_end, delay=timeline.max_length))

        env.run(all_of_event)

    def evaluate(self, configuration: EvaluateSchedule):
        # self.environment.run(self.__run__(configuration))
        pass

    def __run__(self, run_event: simpy.Event, configuration: RunSchedule):
        # TODO: - Verify
        store = simpy.Store(self.environment, capacity=configuration.n_workers)

        def produce():
            for idx, simulation in enumerate(configuration.simulations):
                yield store.put((idx, simulation))

            yield store.put(None)

        def consume():
            while True:
                idx, simulation = yield store.get()

                if simulation is None:
                    break

                simulation.run(idx, self, self, self.environment)

        self.environment.process(produce())

        yield self.environment.process(consume())

        try:
            run_event.succeed()
        except:
            pass

    def __main_timeline__(self, warm_up_event: simpy.Event, run_event: simpy.Event, configuration: TimelineSchedule):
        for index, phase in enumerate(configuration.warm_up_phases):
            self.__update__(WarmUpPhase(index))
            self.logger.info(f'Warm-up phase {index} started')
            yield self.environment.timeout(phase)

        self.logger.info('Warm-up finished')
        self.logger.info('Training started')

        self.__update__(TrainingPhase())

        warm_up_event.succeed()

        yield run_event

        self.logger.info('Training finished')

        self.__update__(EvaluationPhase())

    def __train_timeline__(
        self,
        id: str, configuration: TrainSchedule, warm_up_event: simpy.Event, training_event: simpy.Event,
        run_event: simpy.Event, train_fn: Callable
    ):
        yield warm_up_event

        for i in range(configuration.pretrain_steps):
            train_fn()

        training_steps = configuration.pretrain_steps

        while (training_steps < configuration.max_training_steps
               and not run_event.triggered
               and not training_event.triggered):
            yield self.environment.timeout(configuration.train_interval)

            train_fn()
            training_steps += 1

            if training_steps >= configuration.max_training_steps:
                self.logger.info(f'End training {id} due to max training steps reached')
                break

        try:
            training_event.succeed()
        except:
            pass

        self.logger.info(f'Training finished {id}')

    def __terminate_if_needed__(self, run_event: simpy.Event, delay: float):
        yield self.environment.timeout(float)

        self.logger.info('Terminating simulation due to max duration reached')

        try:
            run_event.succeed()
        except:
            pass

    def __update__(self, phase: Phase):
        self.work_center.update(phase)
        self.machine.update(phase)

    def did_finish_simulation(self, shop_floor_id: int):
        pass
