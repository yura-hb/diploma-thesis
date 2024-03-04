from .simulator import *
from .utils import Queue

from agents.base import Agent


class TDSimulator(Simulator):
    """
    A simulator, which estimates returns in Temporal Difference manner and send information for training as soon as
    possible
    """

    def __init__(self, memory: int = 100, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.memory = memory

        self.machine_queue = Queue(self.machine.is_distributed)
        self.work_center_queue = Queue(self.work_center.is_distributed)

    def did_prepare_machine_record(self, context: Context, machine: Machine, record: Record):
        super().did_prepare_machine_record(context, machine, record)

        self.__store_or_forward_td__(context, self.machine_queue, self.machine, machine.key, record)

    def did_prepare_work_center_record(self, context: Context, work_center: WorkCenter, record: Record):
        super().did_prepare_work_center_record(context, work_center, record)

        self.__store_or_forward_td__(context, self.work_center_queue, self.work_center, work_center.key, record)

    @staticmethod
    def from_cli(parameters, *args, **kwargs) -> Simulator:
        return TDSimulator(parameters.get('memory', 1), *args, **kwargs)

    def __store_or_forward_td__(self, context: Context, queue: Queue, agent: Agent, key, record):
        if self.memory <= 1:
            agent.store(key, record)
            return

        # Implement the idea of n-step memory
        queue.store(context.shop_floor.id, key, context.moment, record)

        if queue.group_len(context.shop_floor.id, key) > self.memory:
            records = queue.pop_group(context.shop_floor.id, key)

            agent.store(key, records)

            queue.store_group(context.shop_floor.id, key, records[1:])

        return
