
from environment import Delegate, Context, MachineKey, WorkCenterKey
from utils import Loggable

from tensordict import TensorDictBase

from simulator.utils import Queue


class MemoryModel(Delegate, Loggable):

    def __init__(self):
        super().__init__()

        self.machine_queue = None
        self.work_center_queue = None

    def connect(self, simulator: 'Simulator'):
        self.machine_queue = Queue(is_distributed=simulator.machine.is_distributed)
        self.work_center_queue = Queue(is_distributed=simulator.work_center.is_distributed)

    def did_start_simulation(self, context: Context):
        pass

    def store_schedule_record(self, context: Context, key: MachineKey, memory: TensorDictBase):
        self.machine_queue.store(context.shop_floor.id, key, 0, memory)

    def get_schedule_record(self, context: Context, key: MachineKey) -> TensorDictBase | None:
        return self.__get__(self.machine_queue, context, key)

    def store_route_record(self, context: Context, key: WorkCenterKey, memory: TensorDictBase):
        self.work_center_queue.store(context.shop_floor.id, key, 0, memory)

    def get_route_record(self, context: Context, key: WorkCenterKey) -> TensorDictBase | None:
        return self.__get__(self.work_center_queue, context, key)

    def did_finish_simulation(self, context: Context):
        self.machine_queue.pop(context.shop_floor.id)
        self.work_center_queue.pop(context.shop_floor.id)

    @staticmethod
    def __get__(queue, context, key):
        value = queue.pop_group(context.shop_floor.id, key)

        if value is None or len(value) == 0:
            return None

        return value[0]
