import logging
from abc import ABCMeta, abstractmethod
from typing import Callable, List

import simpy

from agents import MachineInput, WorkCenterInput
from agents import TrainingPhase, EvaluationPhase, WarmUpPhase, Phase
from agents import Machine as MachineAgent, WorkCenter as WorkCenterAgent
from agents.utils.memory import Record
from environment import Agent, ShopFloor, Job, WaitInfo, Machine, WorkCenter, Context
from tape import TapeModel, SimulatorInterface
from .configuration import RunConfiguration, EvaluateConfiguration
from .simulation import Simulation
from utils import Loggable


def reset_tape():
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            self.tape_model.clear_all()
            result = func(self, *args, **kwargs)
            self.tape_model.clear_all()

            return result

        return wrapper

    return decorator


class Simulator(Agent, Loggable, SimulatorInterface, metaclass=ABCMeta):

    def __init__(self, machine: MachineAgent, work_center: WorkCenterAgent, tape_model: TapeModel):
        super().__init__()

        self.work_center = work_center
        self.machine = machine
        self.tape_model = tape_model

        self.tape_model.connect(self)

    def with_logger(self, logger: logging.Logger):
        super().with_logger(logger)

        self.work_center.with_logger(logger)
        self.machine.with_logger(logger)
        self.tape_model.with_logger(logger)

        return self

    @reset_tape()
    def train(self, environment: simpy.Environment, config: RunConfiguration):
        assert self.machine.is_trainable or self.work_center.is_trainable, 'At least one agent should be trainable'

        env = environment
        warmup_end = env.event()
        machine_training_end = env.event()
        work_center_train_end = env.event()
        run_end = env.event()
        all_of_event = simpy.AllOf(env, [warmup_end, machine_training_end, work_center_train_end, run_end])

        env.process(self.__main_timeline__(
            warm_up_event=warmup_end,
            run_event=run_end,
            configuration=config.timeline,
            environment=env
        ))

        ids = ['machine', 'work_center']
        schedules = [config.machine_train_schedule, config.work_center_train_schedule]
        end_training_events = [machine_training_end, work_center_train_end]
        train_steps = [self.machine.train_step, self.work_center.train_step]

        for idx, schedule, end_of_train_event, train_step in zip(ids, schedules, end_training_events, train_steps):
            env.process(self.__train_timeline__(
                environment=env,
                name=idx,
                configuration=schedule,
                warm_up_event=warmup_end,
                training_event=end_of_train_event,
                run_event=run_end,
                train_fn=train_step
            ))

        env.process(
            self.__run__(
                environment=env,
                run_event=run_end,
                n_workers=config.n_workers,
                simulations=config.simulations,
                is_training=True
            )
        )
        env.process(self.__terminate_if_needed__(
            environment=env,
            run_event=run_end,
            delay=config.timeline.duration
        ))

        env.run(all_of_event)

    @reset_tape()
    def evaluate(self, environment: simpy.Environment, config: EvaluateConfiguration):
        self.__log__(f'Evaluation Started')

        self.__update__(EvaluationPhase())

        run_end = environment.event()

        environment.process(
            self.__run__(
                environment=environment,
                run_event=run_end,
                n_workers=config.n_workers,
                simulations=config.simulations,
                is_training=False
            )
        )

        environment.run(run_end)

        self.__log__(f'Evaluation Ended')

    # Reward

    def encode_machine_state(self, parameters: MachineInput):
        return self.machine.encode_state(parameters)

    def encode_work_center_state(self, parameters: WorkCenterInput):
        return self.work_center.encode_state(parameters)

    @abstractmethod
    def did_prepare_machine_record(self,
                                   shop_floor: ShopFloor,
                                   machine: Machine,
                                   record: Record,
                                   decision_moment: float):
        pass

    @abstractmethod
    def did_prepare_work_center_record(self,
                                       shop_floor: ShopFloor,
                                       work_center: WorkCenter,
                                       record: Record,
                                       decision_moment: float):
        pass

    # Agent

    def schedule(self, context: Context, machine: Machine) -> Job | WaitInfo:
        parameters = MachineInput(machine, context.moment)
        result = self.machine.schedule(parameters)

        if self.machine.is_trainable:
            self.tape_model.register_machine_reward_preparation(context=context,
                                                                machine=machine,
                                                                record=result)

        return result.result

    def route(self, context: Context, work_center: WorkCenter, job: Job) -> 'Machine | None':
        parameters = WorkCenterInput(work_center, job)
        result = self.work_center.schedule(parameters)

        if self.work_center.is_trainable:
            self.tape_model.register_work_center_reward_preparation(context=context,
                                                                    work_center=work_center,
                                                                    job=job,
                                                                    record=result)

        return result.result

    # Timeline

    def __run__(self,
                environment: simpy.Environment,
                run_event: simpy.Event,
                n_workers: int,
                simulations: List[Simulation],
                is_training: bool):
        resource = simpy.Resource(environment, capacity=n_workers)

        def consume(simulation: Simulation):
            with resource.request() as req:
                yield req

                simulation.prepare(self, self.tape_model, environment)

                if is_training:
                    self.tape_model.register(simulation.shop_floor)

                self.__log__(f'Simulation Started {simulation.simulation_id}')

                yield environment.process(simulation.run())

                self.__log__(f'Simulation Finished {simulation.simulation_id}')

        processes = [environment.process(consume(simulation)) for simulation in simulations]

        yield simpy.AllOf(environment, events=processes)

        self.__log__(f'All simulations finished')

        try:
            run_event.succeed()
        except:
            pass

    def __main_timeline__(self,
                          environment: simpy.Environment,
                          warm_up_event: simpy.Event,
                          run_event: simpy.Event,
                          configuration: RunConfiguration.TimelineSchedule):
        for index, phase in enumerate(configuration.warm_up_phases):
            self.__update__(WarmUpPhase(index))
            self.__log__(f'Warm-up phase {index} started')
            yield environment.timeout(phase)

        self.__log__('Warm-up finished')
        self.__log__('Training started')

        self.__update__(TrainingPhase())

        warm_up_event.succeed()

        yield run_event

        self.__log__('Training finished')

        self.__update__(EvaluationPhase())

    def __train_timeline__(
        self,
        environment: simpy.Environment,
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
            yield environment.timeout(configuration.train_interval)

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

    def __terminate_if_needed__(self, environment: simpy.Environment, run_event: simpy.Event, delay: float):
        yield environment.timeout(delay)

        self.__log__('Terminating run due to max duration reached')

        try:
            run_event.succeed()
        except:
            pass

    def __update__(self, phase: Phase):
        self.work_center.update(phase)
        self.machine.update(phase)

    def __log__(self, message: str, level: int = logging.INFO):
        self.logger.log(level=level, msg=message)

