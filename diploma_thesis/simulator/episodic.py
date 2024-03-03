from .simulator import *
from agents.base import Agent
from functools import reduce
from typing import Dict


class EpisodicSimulator(Simulator):
    """
    A simulator, which stores trajectory.yml of agents and emits them for training after it finishes
    """

    class Queue:

        def __init__(self, is_distributed: bool):
            self.is_distributed = is_distributed
            self.queue = dict()

        def store(self, shop_floor_id, key, moment, record):
            self.queue[shop_floor_id] = self.queue.get(shop_floor_id, dict())

            if self.is_distributed:
                self.queue[shop_floor_id][key] = self.queue[shop_floor_id].get(key, dict())
                self.queue[shop_floor_id][key][moment] = self.queue[shop_floor_id][key].get(moment, []) + [record]
            else:
                self.queue[shop_floor_id][moment] = self.queue[shop_floor_id].get(moment, []) + [record]

        def pop(self, shop_floor_id):
            if shop_floor_id not in self.queue:
                return None

            values = self.queue[shop_floor_id]

            del self.queue[shop_floor_id]

            return values

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.machine_queue = self.Queue(self.machine.is_distributed)
        self.work_center_queue = self.Queue(self.work_center.is_distributed)

    def did_prepare_machine_record(self, context: Context, machine: Machine, record: Record):
        super().did_prepare_machine_record(context, machine, record)

        self.machine_queue.store(context.shop_floor.id, machine.key, context.moment, record)

    def did_prepare_work_center_record(self, context: Context, work_center: WorkCenter, record: Record):
        super().did_prepare_work_center_record(context, work_center, record)

        self.work_center_queue.store(context.shop_floor.id, work_center.key, context.moment, record)

    def did_finish_simulation(self, simulation: Simulation):
        super().did_finish_simulation(simulation)

        self.__forward_trajectory__(simulation, self.machine, self.machine_queue)
        self.__forward_trajectory__(simulation, self.work_center, self.work_center_queue)

    @staticmethod
    def from_cli(parameters, *args, **kwargs) -> Simulator:
        return EpisodicSimulator(*args, **kwargs)

    def __forward_trajectory__(self, simulation: Simulation, agent: Agent, queue: Queue):
        records = queue.pop(simulation.shop_floor.id)

        if records is None:
            return

        if agent.is_distributed:
            for key, records in records.items():
                self.__send_trajectory__(key, agent, records)
            return

        self.__send_trajectory__(None, agent, records)

    @staticmethod
    def __send_trajectory__(key, agent: Agent, records: Dict[float, Record]):
        records = sorted(records.items(), key=lambda item: item[0])
        records = reduce(lambda acc, item: acc + item[1], records, [])

        agent.store(key, records)
