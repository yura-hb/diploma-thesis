import weakref
from abc import ABCMeta

from environment import Job, ShopFloor, Machine, DelegateContext
from .simulator_interface import SimulatorInterface

ShopFloorId = str
ActionId = int


class Queue(metaclass=ABCMeta):

    def __init__(self):
        self._simulator = None

    def connect(self, simulator: SimulatorInterface):
        self._simulator = weakref.ref(simulator)

    def prepare(self, shop_floor: ShopFloor):
        pass

    def clear(self, shop_floor: ShopFloor):
        pass

    def record_state(self, context: DelegateContext, machine: Machine, job: Job):
        pass

    def emit_intermediate_reward(self, context: DelegateContext, machine: Machine, job: Job):
        pass

    def emit_reward_after_completion(self, context: DelegateContext, job: Job):
        pass

    @property
    def simulator(self) -> SimulatorInterface:
        return self._simulator()
