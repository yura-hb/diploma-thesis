import logging
import traceback
from abc import ABCMeta, abstractmethod
from dataclasses import field
from typing import Callable, List

import simpy
import torch
from tensordict.prototype import tensorclass

from agents import MachineAgent, WorkCenterAgent
from agents import MachineInput, WorkCenterInput
from agents import TrainingPhase, EvaluationPhase, WarmUpPhase, Phase
from agents.utils.memory import Record
from environment import Agent, Job, Machine, WorkCenter, Context, Delegate
from utils import Loggable
from .configuration import RunConfiguration, EvaluateConfiguration
from .graph import GraphModel
from .memory import MemoryModel
from .simulation import Simulation
from .tape import TapeModel, SimulatorInterface


def reset_tape():
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            self.tape_model.clear_all()
            result = func(self, *args, **kwargs)
            self.tape_model.clear_all()

            return result

        return wrapper

    return decorator


@tensorclass
class RewardCache:
    @tensorclass
    class Record:
        action: torch.Tensor = torch.tensor([])
        reward: torch.Tensor = torch.tensor([])
        moment: torch.Tensor = torch.tensor([])
        work_center_id: torch.Tensor = torch.tensor([])
        entropy: torch.Tensor = torch.tensor([])

    @tensorclass
    class MachineRecord(Record):
        machine_id: torch.Tensor = torch.tensor([])

    @tensorclass
    class WorkCenterRecord(Record):
        work_center_id: torch.Tensor = torch.tensor([])

    machines: MachineRecord = field(default_factory=lambda: RewardCache.MachineRecord(batch_size=[]))
    work_centers: WorkCenterRecord = field(default_factory=lambda: RewardCache.WorkCenterRecord(batch_size=[]))


class Simulator(Agent, Loggable, SimulatorInterface, metaclass=ABCMeta):

    def __init__(self,
                 machine: MachineAgent,
                 work_center: WorkCenterAgent,
                 tape_model: TapeModel,
                 graph_model: GraphModel):
        super().__init__()

        self.work_center = work_center
        self.machine = machine
        self.tape_model = tape_model
        self.tape_model.connect(self)

        self.memory_model = MemoryModel()
        self.memory_model.connect(self)

        self.graph_model = graph_model
        self.cache = dict()

    def with_logger(self, logger: logging.Logger):
        super().with_logger(logger)

        self.work_center.with_logger(logger)
        self.machine.with_logger(logger)
        self.tape_model.with_logger(logger)

        return self

    @reset_tape()
    def train(self,
              environment: simpy.Environment,
              config: RunConfiguration,
              on_simulation_end: Callable[[Simulation], None] = None):
        assert self.machine.is_trainable or self.work_center.is_trainable, 'At least one agent should be trainable'

        self.cache = dict()

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
                is_training=True,
                on_simulation_end=on_simulation_end
            )
        )
        env.process(self.__terminate_if_needed__(
            environment=env,
            run_event=run_end,
            delay=config.timeline.duration
        ))

        environment.process(self.__log_time__(environment=environment, log_tick=config.log_tick))

        env.run(all_of_event)

        reward_record, self.cache = self.cache, None

        return reward_record

    @reset_tape()
    def evaluate(self,
                 environment: simpy.Environment,
                 config: EvaluateConfiguration,
                 on_simulation_end: Callable[[Simulation], None] = None):
        self.__log__(f'Evaluation Started')

        self.__update__(EvaluationPhase())

        run_end = environment.event()

        environment.process(
            self.__run__(
                environment=environment,
                run_event=run_end,
                n_workers=config.n_workers,
                simulations=config.simulations,
                is_training=False,
                on_simulation_end=on_simulation_end
            )
        )

        environment.process(self.__log_time__(environment=environment, log_tick=config.log_tick))

        environment.run(run_end)

        self.__log__(f'Evaluation Ended')

    # Reward

    def encode_machine_state(self, context: Context, machine: Machine, memory):
        parameters = MachineInput(machine=machine,
                                  now=context.moment,
                                  graph=self.graph_model.graph(context=context),
                                  memory=memory)

        return self.machine.encode_state(parameters)

    def encode_work_center_state(self, context: Context, work_center: WorkCenter, memory, job: Job):
        parameters = WorkCenterInput(work_center=work_center,
                                     job=job,
                                     graph=self.graph_model.graph(context=context),
                                     memory=memory)

        return self.work_center.encode_state(parameters)

    @abstractmethod
    def did_prepare_machine_record(self, context: Context, machine: Machine, record: Record):
        if context.shop_floor.name not in self.cache.keys():
            return

        self.cache[context.shop_floor.name].machines = self.__update_cache__(
            self.cache[context.shop_floor.name].machines,
            cls=RewardCache.MachineRecord,
            shop_floor=context.shop_floor,
            record=record,
            decision_moment=context.moment,
            work_center_idx=machine.work_center_idx,
            machine_id=machine.machine_idx
        )

    @abstractmethod
    def did_prepare_work_center_record(self, context: Context, work_center: WorkCenter, record: Record):
        if context.shop_floor.name not in self.cache.keys():
            return

        self.cache[context.shop_floor.name].work_centers = self.__update_cache__(
            self.cache[context.shop_floor.name].work_centers,
            cls=RewardCache.WorkCenterRecord,
            shop_floor=context.shop_floor,
            record=record,
            decision_moment=context.moment,
            work_center_idx=work_center.work_center_idx
        )

    def did_finish_simulation(self, simulation: Simulation):
        pass

    # Agent

    def schedule(self, context: Context, machine: Machine) -> Job | None:
        graph = self.graph_model.graph(context=context)
        memory = self.memory_model.get_schedule_record(context=context, key=machine.key)
        parameters = MachineInput(machine=machine, now=context.moment, graph=graph, memory=memory)

        result = self.machine.schedule(machine.key, parameters)

        if self.machine.is_trainable:
            self.tape_model.register_machine_reward_preparation(context=context, machine=machine, record=result)

        if result.record is not None:
            self.memory_model.store_schedule_record(context=context, key=machine.key, memory=result.record.memory)

        return result.result

    def route(self, context: Context, work_center: WorkCenter, job: Job) -> 'Machine | None':
        graph = self.graph_model.graph(context=context)
        memory = self.memory_model.get_route_record(context=context, key=work_center.key)
        parameters = WorkCenterInput(work_center=work_center, job=job, graph=graph, memory=memory)

        result = self.work_center.schedule(work_center.key, parameters)

        if self.work_center.is_trainable:
            self.tape_model.register_work_center_reward_preparation(context=context,
                                                                    work_center=work_center,
                                                                    job=job,
                                                                    record=result)

        if result.record is not None:
            self.memory_model.store_route_record(context=context, key=work_center.key, memory=result.record.memory)

        return result.result

    # Timeline

    def __run__(self,
                environment: simpy.Environment,
                run_event: simpy.Event,
                n_workers: int,
                simulations: List[Simulation],
                is_training: bool,
                on_simulation_end: Callable):
        resource = simpy.Resource(environment, capacity=n_workers)

        def consume(simulation: Simulation):
            with resource.request() as req:
                is_terminated = False

                try:
                    yield req

                    self.cache[simulation.simulation_id] = RewardCache(batch_size=[])

                    delegate = Delegate([self.tape_model, self.graph_model])

                    simulation.prepare(self, delegate, environment)

                    self.machine.setup(simulation.shop_floor)
                    self.work_center.setup(simulation.shop_floor)

                    if is_training:
                        self.tape_model.register(
                            simulation.shop_floor, self.machine.is_trainable, self.work_center.is_trainable
                        )

                    self.__log__(f'Simulation Started {simulation.simulation_id}')

                    yield environment.process(simulation.run())

                    self.__log__(f'Simulation Finished {simulation.simulation_id}')

                    rewards = self.cache.pop(simulation.simulation_id, RewardCache(batch_size=[]))

                    if is_training:
                        self.did_finish_simulation(simulation)

                    is_terminated = on_simulation_end(simulation, rewards)
                except:
                    self.__log__(f'Skip simulation {simulation.simulation_id} due to error {traceback.print_exc()}')

                if is_terminated:
                    raise ValueError('Terminated')

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

    def __log_time__(self, environment: simpy.Environment, log_tick: float):
        while True:
            self.__log__(f'Timeline {environment.now}')

            yield environment.timeout(log_tick)

            import gc

            gc.collect()

            try:
                torch.mps.empty_cache()
            except:
                pass

            torch.cuda.empty_cache()

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

    @staticmethod
    def __update_cache__(cache, cls, shop_floor, record, decision_moment, work_center_idx, **kwargs):
        if torch.is_tensor(decision_moment):
            decision_moment = decision_moment.view([])
        else:
            decision_moment = torch.tensor(decision_moment)

        policy = record.info[Record.ACTION_KEY]
        entropy = torch.distributions.Categorical(logits=policy).entropy().mean()

        kwargs['entropy'] = entropy

        record = cls(
            action=record.action,
            reward=record.reward,
            moment=decision_moment,
            work_center_id=work_center_idx,
            **kwargs,
            batch_size=[]
        ).view(-1)

        if not cache.batch_size:
            return record

        return torch.cat([cache, record])
