import weakref
from abc import ABCMeta

from agents.machine.model import MachineModel
from agents.workcenter.model import WorkCenterModel
from environment import Job, ShopFloor, Machine, Delegate, WorkCenter, DelegateContext
from .machine import MachineReward, from_cli as machine_reward_from_cli
from .utils import *
from .work_center import WorkCenterReward, from_cli as work_center_reward_from_cli

ShopFloorId = str
# TODO: - Now suppose it to be job id, but should be actual action id
ActionId = int


class TapeModel(Delegate, metaclass=ABCMeta):

    def __init__(self, machine_reward: MachineReward, work_center_reward: WorkCenterReward):
        self.machine_reward = machine_reward
        self.work_center_reward = work_center_reward

        self._simulator = None

        self.machine_queue = MachineQueue(machine_reward)
        self.work_center_queue = WorkCenterQueue(work_center_reward)

    def connect(self, simulator: SimulatorInterface):
        self._simulator = weakref.ref(simulator)

        self.machine_queue.connect(simulator)
        self.work_center_queue.connect(simulator)

    def register_machine_reward_preparation(self, shop_floor: ShopFloor, machine: Machine, record: MachineModel.Record):
        self.machine_queue.register(shop_floor, machine, record)

    def register_work_center_reward_preparation(
        self, shop_floor: ShopFloor, job: Job, work_center: WorkCenter, record: WorkCenterModel.Record
    ):
        self.work_center_queue.register(shop_floor, job, work_center, record)

    def did_start_simulation(self, context: DelegateContext):
        self.machine_queue.prepare(context.shop_floor)
        self.work_center_queue.prepare(context.shop_floor)

    def did_produce(self, context: DelegateContext, job: Job, machine: Machine):
        self.machine_queue.record_state(context, machine, job)
        self.work_center_queue.record_state(context, machine, job)

        self.machine_queue.emit_intermediate_reward(context, machine, job)
        self.work_center_queue.emit_intermediate_reward(context, machine, job)

    def did_complete(self, context: DelegateContext, job: Job):
        self.machine_queue.emit_reward_after_completion(context, job)
        self.work_center_queue.emit_reward_after_completion(context, job)

    def did_finish_simulation(self, context: DelegateContext):
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
