
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

    def get_schedule_record(self, context: Context, key: MachineKey) -> TensorDictBase:
        value = self.machine_queue.pop_group(context.shop_floor.id, key)

        if value is None:
            return value

        return value[0]

    def store_route_record(self, context: Context, key: WorkCenterKey, memory: TensorDictBase):
        self.work_center_queue.store(context.shop_floor.id, key, 0, memory)

    def get_route_record(self, context: Context, key: WorkCenterKey) -> TensorDictBase:
        value = self.work_center_queue.pop_group(context.shop_floor.id, key)

        if value is None:
            return value

        return value[0]

    def did_finish_simulation(self, context: Context):
        self.machine_queue.pop(context.shop_floor.id)
        self.work_center_queue.pop(context.shop_floor.id)
