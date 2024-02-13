import logging
from abc import ABCMeta
from typing import Callable, List, TypeVar

import simpy

from abc import abstractmethod
from agents import WorkCenter, Machine, TrainingPhase, EvaluationPhase, WarmUpPhase, Phase
from environment import Delegate, Agent
from .configuration import RunConfiguration, EvaluateConfiguration
from reward import RewardModel
from .simulation import Simulation


class Simulator(Agent, Delegate, metaclass=ABCMeta):

    def __init__(
        self,
        machine: Machine,
        machine_reward: RewardModel,
        work_center: WorkCenter,
        work_center_reward: RewardModel,
        environment: simpy.Environment,
        logger: logging.Logger
    ):
        self.work_center = work_center
        self.machine = machine
        self.machine_reward = machine_reward
        self.work_center_reward = work_center_reward
        self.environment = environment
        self.logger = logger

        self.__post_init__()

    @abstractmethod
    def __post_init__(self):
        pass

    def train(self, config: RunConfiguration):
        # TODO: - Revert
        # assert self.machine.is_trainable or self.work_center.is_trainable, 'At least one agent should be trainable'

        env = self.environment
        warmup_end = env.event()
        machine_training_end = env.event()
        work_center_train_end = env.event()
        run_end = env.event()
        all_of_event = simpy.AllOf(env, [warmup_end, machine_training_end, work_center_train_end, run_end])

        env.process(self.__main_timeline__(warm_up_event=warmup_end, run_event=run_end, configuration=config.timeline))

        ids = ['machine', 'work_center']
        schedules = [config.machine_train_schedule, config.work_center_train_schedule]
        end_training_events = [machine_training_end, work_center_train_end]
        train_steps = [self.machine.train_step, self.work_center.train_step]

        for idx, schedule, end_of_train_event, train_step in zip(ids, schedules, end_training_events, train_steps):
            env.process(self.__train_timeline__(idx, schedule, warmup_end, end_of_train_event, run_end, train_step))

        env.process(self.__run__(run_event=run_end, n_workers=config.n_workers, simulations=config.simulations))
        env.process(self.__terminate_if_needed__(run_event=run_end, delay=config.timeline.duration))

        env.run(all_of_event)

    def evaluate(self, configuration: EvaluateConfiguration):
        self.__log__(f'Evaluation Started')

        self.__update__(EvaluationPhase())

        run_end = self.environment.event()

        self.environment.process(
            self.__run__(run_end, n_workers=configuration.n_workers, simulations=configuration.simulations)
        )

        self.environment.run(run_end)

        self.__log__(f'Evaluation Ended')

    def __run__(self, run_event: simpy.Event, n_workers: int, simulations: List[Simulation]):
        resource = simpy.Resource(self.environment, capacity=n_workers)

        def consume(simulation: Simulation):
            with resource.request() as req:
                yield req

                self.__log__(f'Simulation Started {simulation.shop_floor_id}')

                yield self.environment.process(simulation.run(self, self, self.environment))

                self.__log__(f'Simulation Finished {simulation.shop_floor_id}')

        processes = [self.environment.process(consume(simulation)) for simulation in simulations]

        yield simpy.AllOf(self.environment, events=processes)

        self.__log__(f'All simulations finished')

        try:
            run_event.succeed()
        except:
            pass

    def __main_timeline__(self,
                          warm_up_event: simpy.Event,
                          run_event: simpy.Event,
                          configuration: RunConfiguration.TimelineSchedule):
        for index, phase in enumerate(configuration.warm_up_phases):
            self.__update__(WarmUpPhase(index))
            self.__log__(f'Warm-up phase {index} started')
            yield self.environment.timeout(phase)

        self.__log__('Warm-up finished')
        self.__log__('Training started')

        self.__update__(TrainingPhase())

        warm_up_event.succeed()

        yield run_event

        self.__log__('Training finished')

        self.__update__(EvaluationPhase())

    def __train_timeline__(
        self,
        name: str,
        configuration: RunConfiguration.TrainSchedule,
        warm_up_event: simpy.Event,
        training_event: simpy.Event,
        run_event: simpy.Event,
        train_fn: Callable
    ):
        yield warm_up_event

        for i in range(configuration.pretrain_steps):
            train_fn()

        training_steps = configuration.pretrain_steps

        while (training_steps < configuration.max_training_steps
               and not run_event.triggered
               and not training_event.triggered):
            yield self.environment.timeout(configuration.train_interval)

            self.__log__(f'Training Step {training_steps} at {name} ', logging.DEBUG)

            train_fn()
            training_steps += 1

            if training_steps >= configuration.max_training_steps:
                self.__log__(f'End training {name} due to max training steps reached')
                break

        try:
            training_event.succeed()
        except:
            pass

        self.__log__(f'Training finished {name}')

    def __terminate_if_needed__(self, run_event: simpy.Event, delay: float):
        yield self.environment.timeout(delay)

        self.__log__('Terminating simulation due to max duration reached')

        try:
            run_event.succeed()
        except:
            pass

    def __update__(self, phase: Phase):
        self.work_center.update(phase)
        self.machine.update(phase)

    def __log__(self, message: str, level: int = logging.INFO):
        self.logger.log(level=level, msg=message)

