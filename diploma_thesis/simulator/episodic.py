from .simulator import *
from agents.base import Agent
from agents.base.agent import Trajectory
from functools import reduce
from typing import Dict
from .utils import Queue


class EpisodicSimulator(Simulator):
    """
    A simulator, which stores trajectory.yml of agents and emits them for training after it finishes
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.machine_queue = Queue(self.machine.is_distributed)
        self.work_center_queue = Queue(self.work_center.is_distributed)

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
                self.__send_trajectory__(simulation.shop_floor.id, key, agent, records)
            return

        self.__send_trajectory__(simulation.shop_floor.id, None, agent, records)

    @staticmethod
    def __send_trajectory__(episode_id: int, key, agent: Agent, records: Dict[float, Record]):
        records = sorted(records.items(), key=lambda item: item[0])
        records = reduce(lambda acc, item: acc + item[1], records, [])

        if len(records) > 0:
            records[-1].done = torch.tensor(True, dtype=torch.bool)

        agent.store(key, Trajectory(episode_id=episode_id, records=records))
