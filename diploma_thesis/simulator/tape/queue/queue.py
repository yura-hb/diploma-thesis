import weakref
from abc import ABCMeta
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

from agents.utils.memory import Record
from environment import Job, ShopFloor, Machine
from simulator.tape.utils.simulator_interface import SimulatorInterface
from utils import Loggable

ShopFloorId = int
ActionId = int


class NextStateRecordMode(StrEnum):
    on_produce = 'on_produce'
    on_next_action = 'on_next_action'


Context = TypeVar('Context')


@dataclass
class TapeRecord:
    job_id: int | None
    record: Record
    context: Context
    moment: int
    mode: NextStateRecordMode = NextStateRecordMode.on_produce


class Queue(Loggable, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()
        self._simulator = None

    def connect(self, simulator: SimulatorInterface):
        self._simulator = weakref.ref(simulator)

    def prepare(self, shop_floor: ShopFloor):
        pass

    def clear(self, shop_floor: ShopFloor):
        pass

    def did_produce(self, context: Context, machine: Machine, job: Job):
        pass

    def did_complete(self, context: Context, job: Job):
        pass

    @property
    def simulator(self) -> SimulatorInterface:
        return self._simulator()
