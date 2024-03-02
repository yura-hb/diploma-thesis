import logging
import weakref
from abc import ABCMeta

from agents.machine.model import MachineModel
from agents.workcenter.model import WorkCenterModel
from environment import Job, ShopFloor, Machine, Delegate, WorkCenter, Context
from .machine import MachineReward, from_cli as machine_reward_from_cli
from .work_center import WorkCenterReward, from_cli as work_center_reward_from_cli
from .queue import MachineQueue, WorkCenterQueue, NextStateRecordMode
from .utils import *
from utils import Loggable, filter


class TapeModel(Delegate, Loggable, metaclass=ABCMeta):

    def __init__(self,
                 machine_reward: MachineReward,
                 work_center_reward: WorkCenterReward,
                 mode: NextStateRecordMode = NextStateRecordMode.on_produce):
        super().__init__()

        self.machine_reward = machine_reward
        self.work_center_reward = work_center_reward
        self.next_state_record_mode = mode

        self._simulator = None

        self.registered_shop_floor_ids = set()
        self._machine_queue = dict()
        self._work_center_queue = dict()

    def with_logger(self, logger: logging.Logger):
        super().with_logger(logger)

        _ = {v.with_logger(logger) for _, v in self._machine_queue.items()}
        _ = {v.with_logger(logger) for _, v in self._work_center_queue.items()}

        return self

    def connect(self, simulator: SimulatorInterface):
        self._simulator = weakref.ref(simulator)

        _ = {v.connect(simulator) for _, v in self._machine_queue.items()}
        _ = {v.connect(simulator) for _, v in self._work_center_queue.items()}

    def register(self, shop_floor: ShopFloor, is_machine_trainable: bool, is_work_center_trainable: bool):
        self.registered_shop_floor_ids.add(shop_floor.id)

        stores = [self._machine_queue, self._work_center_queue]
        rewards = [self.machine_reward, self.work_center_reward]
        is_trainable = [is_machine_trainable, is_work_center_trainable]
        queues = [MachineQueue, WorkCenterQueue]

        for store, reward, is_trainable, queue in zip(stores, rewards, is_trainable, queues):
            if not is_trainable:
                continue

            q = queue(reward)
            q.with_logger(self.logger)
            q.connect(self.simulator)
            q.prepare(shop_floor)

            store[shop_floor.id] = q

    def clear_all(self):
        self.registered_shop_floor_ids.clear()

        self._machine_queue.clear()
        self._work_center_queue.clear()

    @filter(lambda self, context, *args, **kwargs: context.shop_floor.id in self.registered_shop_floor_ids)
    def register_machine_reward_preparation(self, context: Context, machine: Machine, record: MachineModel.Record):
        if queue := self.machine_queue(context):
            queue.register(context, machine, record, self.next_state_record_mode)

    @filter(lambda self, context, *args, **kwargs: context.shop_floor.id in self.registered_shop_floor_ids)
    def register_work_center_reward_preparation(
         self, context: Context, work_center: WorkCenter, job: Job, record: WorkCenterModel.Record
    ):
        if queue := self.work_center_queue(context):
            queue.register(context, work_center, job, record, self.next_state_record_mode)

    @filter(lambda self, context, *args, **kwargs: context.shop_floor.id in self.registered_shop_floor_ids)
    def did_produce(self, context: Context, job: Job, machine: Machine):
        if queue := self.machine_queue(context):
            queue.did_produce(context, machine, job)

        if queue := self.work_center_queue(context):
            queue.did_produce(context, machine, job)

    @filter(lambda self, context, *args, **kwargs: context.shop_floor.id in self.registered_shop_floor_ids)
    def did_complete(self, context: Context, job: Job):
        if queue := self.machine_queue(context):
            queue.did_complete(context, job)

        if queue := self.work_center_queue(context):
            queue.did_complete(context, job)

    @filter(lambda self, context, *args, **kwargs: context.shop_floor.id in self.registered_shop_floor_ids)
    def did_finish_simulation(self, context: Context):
        self.registered_shop_floor_ids.remove(context.shop_floor.id)

        if context.shop_floor.id in self._machine_queue:
            self._machine_queue.pop(context.shop_floor.id)

        if context.shop_floor.id in self._work_center_queue:
            self._work_center_queue.pop(context.shop_floor.id)

    # Utils

    def machine_queue(self, context: Context) -> MachineQueue | None:
        return self._machine_queue.get(context.shop_floor.id)

    def work_center_queue(self, context: Context) -> WorkCenterQueue | None:
        return self._work_center_queue.get(context.shop_floor.id)

    @property
    def simulator(self) -> SimulatorInterface:
        return self._simulator()

    # CLI

    @staticmethod
    def from_cli(parameters: dict):
        machine_reward = machine_reward_from_cli(parameters['machine_reward'])
        work_center_reward = work_center_reward_from_cli(parameters['work_center_reward'])
        next_state_record_mode = NextStateRecordMode(parameters.get('next_state_record_mode', 'on_produce'))

        return TapeModel(machine_reward, work_center_reward, next_state_record_mode)
