from agents.base.agent import Slice, Trajectory
from .simulator import *
from .utils import Queue


class TDSimulator(Simulator):
    """
    A simulator, which estimates returns in Temporal Difference manner and send information for training as soon as
    possible
    """

    def __init__(self, memory: int = 100, emit_trajectory: bool = False, reset_trajectory: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.memory = memory
        self.emit_trajectory = emit_trajectory
        self.reset_trajectory = reset_trajectory
        self.episode = 0

        self.machine_queue = Queue(self.machine.is_distributed)
        self.work_center_queue = Queue(self.work_center.is_distributed)

    def did_prepare_machine_record(self, context: Context, machine: Machine, record: Record):
        super().did_prepare_machine_record(context, machine, record)

        self.__store_or_forward_td__(context, self.machine_queue, self.machine, machine.key, record)

    def did_prepare_work_center_record(self, context: Context, work_center: WorkCenter, record: Record):
        super().did_prepare_work_center_record(context, work_center, record)

        self.__store_or_forward_td__(context, self.work_center_queue, self.work_center, work_center.key, record)

    def __store_or_forward_td__(self, context: Context, queue: Queue, agent, key, record):
        if self.memory <= 1:
            agent.store(key, Slice(episode_id=context.shop_floor.id, records=[record]))
            return

        queue.store(context.shop_floor.id, key, context.moment, record)

        if queue.group_len(context.shop_floor.id, key) > self.memory:
            # Pass a copy of records to avoid modification of the original
            original_records = queue.pop_group(context.shop_floor.id, key)
            records = [record.clone() for record in original_records]

            if self.emit_trajectory:
                agent.store(key, Trajectory(episode_id=self.episode, records=records))

                self.episode += 1
            else:
                agent.store(key, Slice(episode_id=context.shop_floor.id, records=records))

            if not self.emit_trajectory or not self.reset_trajectory:
                queue.store_group(context.shop_floor.id, key, original_records[1:])

        return

    @staticmethod
    def from_cli(parameters, *args, **kwargs) -> Simulator:
        return TDSimulator(parameters.get('memory', 1), 
                           parameters.get('emit_trajectory', False),
                           parameters.get('reset_trajectory', True),
                           *args, **kwargs)