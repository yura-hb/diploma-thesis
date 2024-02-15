import logging
import weakref
from abc import ABCMeta

from agents.machine.model import MachineModel
from agents.workcenter.model import WorkCenterModel
from environment import Job, ShopFloor, Machine, Delegate, WorkCenter, Context
from .machine import MachineReward, from_cli as machine_reward_from_cli
from .work_center import WorkCenterReward, from_cli as work_center_reward_from_cli
from .queue import MachineQueue, WorkCenterQueue
from .utils import *
from utils import Loggable, filter


class TapeModel(Delegate, Loggable, metaclass=ABCMeta):

    def __init__(self, machine_reward: MachineReward, work_center_reward: WorkCenterReward):
        super().__init__()

        self.machine_reward = machine_reward
        self.work_center_reward = work_center_reward

        self._simulator = None

        self.registered_shop_floor_ids = set()
        self.machine_queue = MachineQueue(machine_reward)
        self.work_center_queue = WorkCenterQueue(work_center_reward)

    def with_logger(self, logger: logging.Logger):
        super().with_logger(logger)

        self.machine_queue.with_logger(logger)
        self.work_center_queue.with_logger(logger)

        return self

    def connect(self, simulator: SimulatorInterface):
        self._simulator = weakref.ref(simulator)

        self.machine_queue.connect(simulator)
        self.work_center_queue.connect(simulator)

    def register(self, shop_floor: ShopFloor):
        self.registered_shop_floor_ids.add(shop_floor.id)

        self.machine_queue.prepare(shop_floor)
        self.work_center_queue.prepare(shop_floor)

    def clear_all(self):
        self.registered_shop_floor_ids.clear()

        self.machine_queue.clear_all()
        self.work_center_queue.clear_all()

    @filter(lambda self, context, *args, **kwargs: context.shop_floor.id in self.registered_shop_floor_ids)
    def register_machine_reward_preparation(self, context: Context, machine: Machine, record: MachineModel.Record):
        self.machine_queue.register(context, machine, record)

    @filter(lambda self, context, *args, **kwargs: context.shop_floor.id in self.registered_shop_floor_ids)
    def register_work_center_reward_preparation(
        self, context: Context, work_center: WorkCenter, job: Job, record: WorkCenterModel.Record
    ):
        self.work_center_queue.register(context, work_center,  job, record)

    @filter(lambda self, context, *args, **kwargs: context.shop_floor.id in self.registered_shop_floor_ids)
    def did_produce(self, context: Context, job: Job, machine: Machine):
        self.machine_queue.record_next_state(context, machine, job)
        self.work_center_queue.record_next_state(context, machine, job)

        self.machine_queue.emit_intermediate_reward(context, machine, job)
        self.work_center_queue.emit_intermediate_reward(context, machine, job)

    @filter(lambda self, context, *args, **kwargs: context.shop_floor.id in self.registered_shop_floor_ids)
    def did_complete(self, context: Context, job: Job):
        self.machine_queue.emit_reward_after_completion(context, job)
        self.work_center_queue.emit_reward_after_completion(context, job)

    @filter(lambda self, context, *args, **kwargs: context.shop_floor.id in self.registered_shop_floor_ids)
    def did_finish_simulation(self, context: Context):
        self.registered_shop_floor_ids.remove(context.shop_floor.id)

        self.machine_queue.clear(context.shop_floor)
        self.work_center_queue.clear(context.shop_floor)

    # Utils

    @property
    def simulator(self) -> SimulatorInterface:
        return self._simulator()

    # CLI

    @staticmethod
    def from_cli(parameters: dict):
        machine_reward = machine_reward_from_cli(parameters['machine_reward'])
        work_center_reward = work_center_reward_from_cli(parameters['work_center_reward'])

        return TapeModel(machine_reward, work_center_reward)
